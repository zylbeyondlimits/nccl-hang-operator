package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"time"

	"github.com/zylbeyondlimits/nccl-hang-operator/api/v1alpha1"
	"github.com/zylbeyondlimits/nccl-hang-operator/internal/checkpoint"
	"github.com/zylbeyondlimits/nccl-hang-operator/internal/telemetry"
)

// TelemetryStore provides the latest rank heartbeats streamed from the node-agent DaemonSet.
type TelemetryStore interface {
	LatestByJob(ctx context.Context, jobUID string) ([]telemetry.RankSample, error)
}

// ClusterClient abstracts Kubernetes control-plane mutations so the reconciler is fully testable.
type ClusterClient interface {
	CordonAndTaintNode(ctx context.Context, nodeName, taintKey, taintValue, taintEffect string) error
	PatchPyTorchJobResume(ctx context.Context, namespace, name, immutableCkptPath, warmSpareLabel string) error
	LaunchDCGMDiagJob(ctx context.Context, nodeName string, level int) error
}

// IncidentPayload is POSTed to the Devin Bot webhook for automated RCA & PR generation.
type IncidentPayload struct {
	JobName              string                 `json:"job_name"`
	JobUID               string                 `json:"job_uid"`
	CulpritNode          string                 `json:"culprit_node"`
	CulpritRank          int32                  `json:"culprit_rank"`
	Reason               string                 `json:"reason"`
	VerifiedCheckpoint   string                 `json:"verified_checkpoint"`
	VerifiedStep         int64                  `json:"verified_step"`
	StepsLost            int64                  `json:"steps_lost"`
	EstimatedGPUHoursSav float64                `json:"estimated_gpu_hours_saved"`
	RankTraceDiff        []telemetry.RankSample `json:"rank_trace_diff"`
	DetectedAt           time.Time              `json:"detected_at"`
}

// WebhookNotifier sends incident bundles to the Devin remediation bot.
type WebhookNotifier interface {
	Notify(ctx context.Context, url string, payload IncidentPayload) error
}

// HTTPWebhookNotifier delivers JSON incident payloads over HTTP.
type HTTPWebhookNotifier struct {
	Client *http.Client
}

func (n *HTTPWebhookNotifier) Notify(ctx context.Context, url string, payload IncidentPayload) error {
	if url == "" {
		return nil
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	cli := n.Client
	if cli == nil {
		cli = &http.Client{Timeout: 5 * time.Second}
	}
	resp, err := cli.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("webhook returned status %d", resp.StatusCode)
	}
	return nil
}

// ReconcileResult mirrors controller-runtime's reconcile return contract.
type ReconcileResult struct {
	RequeueAfter time.Duration
}

// TrainingHealthReconciler drives the Healthy -> Suspect -> Confirming -> Quarantining -> Recovering FSM.
type TrainingHealthReconciler struct {
	Telemetry TelemetryStore
	Cluster   ClusterClient
	Notifier  WebhookNotifier
	Now       func() time.Time
}

func (r *TrainingHealthReconciler) now() time.Time {
	if r.Now != nil {
		return r.Now()
	}
	return time.Now().UTC()
}

// Reconcile evaluates a single TrainingHealthPolicy tick.
func (r *TrainingHealthReconciler) Reconcile(ctx context.Context, policy *v1alpha1.TrainingHealthPolicy) (ReconcileResult, error) {
	now := r.now()
	if policy.Status.State == "" {
		policy.Status.State = v1alpha1.StateHealthy
		policy.Status.LastTransitionTime = now
	}

	samples, err := r.Telemetry.LatestByJob(ctx, policy.UID)
	if err != nil {
		return ReconcileResult{RequeueAfter: 10 * time.Second}, err
	}

	culprit, reason := r.detectAndIsolate(policy, samples, now)

	switch policy.Status.State {
	case v1alpha1.StateHealthy:
		if culprit == nil {
			return ReconcileResult{RequeueAfter: 10 * time.Second}, nil
		}
		t := now
		policy.Status.State = v1alpha1.StateSuspect
		policy.Status.SuspectSince = &t
		policy.Status.CulpritNode = culprit.NodeName
		policy.Status.CulpritRank = culprit.Rank
		policy.Status.CulpritReason = reason
		policy.Status.LastTransitionTime = now
		return ReconcileResult{RequeueAfter: 5 * time.Second}, nil

	case v1alpha1.StateSuspect:
		if culprit == nil {
			r.resetHealthy(policy, now)
			return ReconcileResult{RequeueAfter: 10 * time.Second}, nil
		}
		t := now
		policy.Status.State = v1alpha1.StateConfirming
		policy.Status.ConfirmingSince = &t
		policy.Status.CulpritNode = culprit.NodeName
		policy.Status.CulpritRank = culprit.Rank
		policy.Status.CulpritReason = reason
		policy.Status.LastTransitionTime = now
		return ReconcileResult{RequeueAfter: policy.Spec.ConfirmWindow}, nil

	case v1alpha1.StateConfirming:
		if culprit == nil {
			// False-positive backoff: I/O or SM activity resumed before confirm window elapsed.
			r.resetHealthy(policy, now)
			return ReconcileResult{RequeueAfter: 10 * time.Second}, nil
		}
		if policy.Status.ConfirmingSince != nil && now.Sub(*policy.Status.ConfirmingSince) < policy.Spec.ConfirmWindow {
			return ReconcileResult{RequeueAfter: 5 * time.Second}, nil
		}

		// Confirmed hang/fault -> Quarantine culprit node, verify checkpoint, notify Devin Bot.
		policy.Status.State = v1alpha1.StateQuarantining
		policy.Status.CulpritNode = culprit.NodeName
		policy.Status.CulpritRank = culprit.Rank
		policy.Status.CulpritReason = reason
		policy.Status.LastTransitionTime = now

		rem := policy.Spec.Remediation
		if err := r.Cluster.CordonAndTaintNode(ctx, culprit.NodeName, rem.TaintKey, "nccl-hang", rem.TaintEffect); err != nil {
			return ReconcileResult{RequeueAfter: 5 * time.Second}, fmt.Errorf("cordon/taint %s: %w", culprit.NodeName, err)
		}

		verified, err := checkpoint.SelectVerified(rem.CheckpointRoot, policy.Spec.TargetRef.WorldSize)
		if err != nil {
			return ReconcileResult{RequeueAfter: 10 * time.Second}, fmt.Errorf("verify checkpoint: %w", err)
		}

		// Default 30m NCCL watchdog minus ~1m fast detection across WorldSize GPUs.
		gpuHoursSaved := float64(policy.Spec.TargetRef.WorldSize) * (29.0 / 60.0)
		policy.Status.LastVerifiedCkptPath = verified.Path
		policy.Status.LastVerifiedCkptStep = verified.Step
		policy.Status.StepsLost = verified.StepsLost
		policy.Status.GPUHoursSaved += gpuHoursSaved

		if r.Notifier != nil && rem.WebhookURL != "" {
			_ = r.Notifier.Notify(ctx, rem.WebhookURL, IncidentPayload{
				JobName:              policy.Spec.TargetRef.Name,
				JobUID:               policy.UID,
				CulpritNode:          culprit.NodeName,
				CulpritRank:          culprit.Rank,
				Reason:               reason,
				VerifiedCheckpoint:   verified.Path,
				VerifiedStep:         verified.Step,
				StepsLost:            verified.StepsLost,
				EstimatedGPUHoursSav: gpuHoursSaved,
				RankTraceDiff:        samples,
				DetectedAt:           now,
			})
		}

		// Transition to Recovering: patch PyTorchJob with immutable checkpoint path & run DCGM diag.
		if err := r.Cluster.PatchPyTorchJobResume(ctx, policy.Spec.TargetRef.Namespace, policy.Spec.TargetRef.Name, verified.Path, rem.WarmSpareLabel); err != nil {
			return ReconcileResult{RequeueAfter: 5 * time.Second}, fmt.Errorf("patch PyTorchJob: %w", err)
		}
		if err := r.Cluster.LaunchDCGMDiagJob(ctx, culprit.NodeName, rem.DCGMDiagLevel); err != nil {
			return ReconcileResult{RequeueAfter: 10 * time.Second}, fmt.Errorf("launch DCGM diag: %w", err)
		}

		policy.Status.State = v1alpha1.StateRecovering
		policy.Status.LastTransitionTime = now
		return ReconcileResult{RequeueAfter: 15 * time.Second}, nil

	case v1alpha1.StateQuarantining, v1alpha1.StateRecovering:
		if culprit == nil {
			r.resetHealthy(policy, now)
		}
		return ReconcileResult{RequeueAfter: 15 * time.Second}, nil
	}

	return ReconcileResult{RequeueAfter: 10 * time.Second}, nil
}

func (r *TrainingHealthReconciler) resetHealthy(policy *v1alpha1.TrainingHealthPolicy, now time.Time) {
	policy.Status.State = v1alpha1.StateHealthy
	policy.Status.SuspectSince = nil
	policy.Status.ConfirmingSince = nil
	policy.Status.LastTransitionTime = now
}

// hasActiveRDMAProgress checks whether NIC ASIC counters (port_xmit_data / port_rcv_data)
// show forward data movement with zero fabric/HW errors, distinguishing a legitimate slow
// cross-switch InfiniBand/RoCE collective straggler from a true zero-progress deadlock.
func hasActiveRDMAProgress(spec v1alpha1.TrainingHealthPolicySpec, s telemetry.RankSample) bool {
	if s.IBErrDelta > 0 || s.RDMAHWErrorDelta > 0 {
		return false
	}
	return spec.MinRDMABytesPerSec > 0 && s.RDMAProgressBytesPerSec >= spec.MinRDMABytesPerSec
}

// isLegitimatePause prevents false-positive kills during eval loops, active Lustre/NFS/S3 checkpoint
// flushes, or slow-yet-progressing cross-switch RDMA collectives.
func isLegitimatePause(spec v1alpha1.TrainingHealthPolicySpec, s telemetry.RankSample, now time.Time) bool {
	if s.OpenCollective != nil && !s.OpenCollective.Completed {
		return hasActiveRDMAProgress(spec, s) // slow fabric moving bytes -> back off; zero bytes -> hang
	}
	if s.SMUtilPct >= spec.SMUtilFloorPct {
		return true // GPU still executing forward passes (e.g., eval loop)
	}
	if s.ProcWriteBytesPerSec >= spec.MinDiskIOBytesPerSec {
		return true // actively flushing checkpoint bytes to storage
	}
	if (s.Phase == "eval" || s.Phase == "ckpt") && now.Before(s.PhaseLeaseUntil) {
		return true // bounded phase lease still valid
	}
	return false
}

// detectAndIsolate checks hardware/fabric faults first, then diffs Flight Recorder collective seq_ids
// across ranks to pinpoint the exact deadlocked rank rather than innocent waiting ranks.
func (r *TrainingHealthReconciler) detectAndIsolate(
	policy *v1alpha1.TrainingHealthPolicy,
	samples []telemetry.RankSample,
	now time.Time,
) (*telemetry.RankSample, string) {
	if len(samples) == 0 {
		return nil, ""
	}

	fatalSet := make(map[int32]bool, len(policy.Spec.FatalXids))
	for _, code := range policy.Spec.FatalXids {
		fatalSet[code] = true
	}

	// 1. Hardware/fabric faults immediately identify the culprit node.
	for i := range samples {
		s := samples[i]
		for _, xid := range s.FatalXids {
			if fatalSet[xid] {
				return &samples[i], fmt.Sprintf("fatal NVML Xid %d on rank %d (%s)", xid, s.Rank, s.NodeName)
			}
		}
		if s.IBErrDelta > 0 {
			return &samples[i], fmt.Sprintf("IB/RDMA error delta=%d on rank %d (%s)", s.IBErrDelta, s.Rank, s.NodeName)
		}
		if s.RDMAHWErrorDelta > 0 {
			return &samples[i], fmt.Sprintf("ConnectX RDMA hw_counter error delta=%d on rank %d (%s)", s.RDMAHWErrorDelta, s.Rank, s.NodeName)
		}
	}

	// 2. If any rank is actively moving RDMA bytes on the fabric with zero IB/HW/Xid errors,
	// a slow cross-switch collective (e.g., AllGather straggler) is still making forward progress.
	var anyActiveRDMA bool
	for i := range samples {
		if hasActiveRDMAProgress(policy.Spec, samples[i]) {
			anyActiveRDMA = true
			break
		}
	}

	// 3. Detect open unfinished NCCL collective past DetectWindow, or wedged non-collective phase past lease.
	var openCollectiveStuck bool
	var wedgedRank *telemetry.RankSample
	for i := range samples {
		s := samples[i]
		if s.OpenCollective != nil && !s.OpenCollective.Completed && now.Sub(s.OpenCollective.StartedAt) >= policy.Spec.DetectWindow {
			if !anyActiveRDMA {
				openCollectiveStuck = true
			}
		}
		if !anyActiveRDMA && !isLegitimatePause(policy.Spec, s, now) && s.VRAMUsedMB > 0 {
			if wedgedRank == nil || s.LastCompletedSeqID < wedgedRank.LastCompletedSeqID {
				wedgedRank = &samples[i]
			}
		}
	}

	if !openCollectiveStuck && wedgedRank == nil {
		return nil, ""
	}

	// 4. Diff Flight Recorder seq_ids across all ranks: the rank with the minimum LastCompletedSeqID
	// (or the one without an open collective while peers wait inside AllReduce) is the true culprit.
	var minSeq uint64 = math.MaxUint64
	var culprit *telemetry.RankSample
	for i := range samples {
		s := samples[i]
		if s.LastCompletedSeqID < minSeq {
			minSeq = s.LastCompletedSeqID
			culprit = &samples[i]
		}
	}
	if openCollectiveStuck && culprit != nil {
		return culprit, fmt.Sprintf("stuck NCCL collective ring: lagging rank %d (seq_id=%d) on %s", culprit.Rank, culprit.LastCompletedSeqID, culprit.NodeName)
	}
	if wedgedRank != nil {
		return wedgedRank, fmt.Sprintf("wedged training/checkpoint phase past lease (SM=%.1f%%, io=%d B/s) on rank %d (%s)",
			wedgedRank.SMUtilPct, wedgedRank.ProcWriteBytesPerSec, wedgedRank.Rank, wedgedRank.NodeName)
	}
	return nil, ""
}
