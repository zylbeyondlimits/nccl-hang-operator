package controller

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/zylbeyondlimits/nccl-hang-operator/api/v1alpha1"
	"github.com/zylbeyondlimits/nccl-hang-operator/internal/checkpoint"
	"github.com/zylbeyondlimits/nccl-hang-operator/internal/telemetry"
)

type fakeTelemetryStore struct {
	samples []telemetry.RankSample
}

func (f *fakeTelemetryStore) LatestByJob(_ context.Context, _ string) ([]telemetry.RankSample, error) {
	return f.samples, nil
}

type fakeClusterClient struct {
	cordonedNodes []string
	patchedCkpt   string
	patchedPool   string
	diagNodes     []string
}

func (f *fakeClusterClient) CordonAndTaintNode(_ context.Context, nodeName, _, _, _ string) error {
	f.cordonedNodes = append(f.cordonedNodes, nodeName)
	return nil
}

func (f *fakeClusterClient) PatchPyTorchJobResume(_ context.Context, _, _, immutableCkptPath, warmSpareLabel string) error {
	f.patchedCkpt = immutableCkptPath
	f.patchedPool = warmSpareLabel
	return nil
}

func (f *fakeClusterClient) LaunchDCGMDiagJob(_ context.Context, nodeName string, _ int) error {
	f.diagNodes = append(f.diagNodes, nodeName)
	return nil
}

type fakeWebhook struct {
	events []IncidentPayload
}

func (f *fakeWebhook) Notify(_ context.Context, _ string, payload IncidentPayload) error {
	f.events = append(f.events, payload)
	return nil
}

func seedCheckpoints(t *testing.T, worldSize int, corruptLatest bool) (string, string) {
	t.Helper()
	root := t.TempDir()
	goodDir := writeFixtureStep(t, root, 500, worldSize, false)
	if corruptLatest {
		_ = writeFixtureStep(t, root, 600, worldSize, true)
	}
	return root, goodDir
}

func writeFixtureStep(t *testing.T, root string, step int64, worldSize int, corrupt bool) string {
	t.Helper()
	dir := filepath.Join(root, fmt.Sprintf("step-%d", step))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	shards := make(map[string]string, worldSize)
	for r := 0; r < worldSize; r++ {
		name := fmt.Sprintf("rank-%d.safetensors", r)
		p := filepath.Join(dir, name)
		header := `{"w":{"dtype":"BF16","shape":[16],"data_offsets":[0,32]}}`
		f, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		_ = binary.Write(f, binary.LittleEndian, uint64(len(header)))
		_, _ = f.WriteString(header)
		payload := make([]byte, 32)
		if corrupt && r == worldSize-1 {
			payload = make([]byte, 8) // truncated write
		}
		_, _ = f.Write(payload)
		_ = f.Close()

		raw, _ := os.ReadFile(p)
		sum := sha256.Sum256(raw)
		shards[name] = hex.EncodeToString(sum[:])
	}
	m := checkpoint.Manifest{
		Step:           step,
		WorldSize:      worldSize,
		OptimizerStep:  step,
		DataloaderStep: step,
		Shards:         shards,
	}
	raw, _ := json.Marshal(m)
	_ = os.WriteFile(filepath.Join(dir, ".metadata.json"), raw, 0o644)
	return dir
}

func basePolicy(ckptRoot string, worldSize int) *v1alpha1.TrainingHealthPolicy {
	return &v1alpha1.TrainingHealthPolicy{
		APIVersion: "nccl.infra.io/v1alpha1",
		Kind:       "TrainingHealthPolicy",
		Namespace:  "ml-train",
		Name:       "llama-pretrain-health",
		UID:        "job-uid-001",
		Spec: v1alpha1.TrainingHealthPolicySpec{
			TargetRef: v1alpha1.TargetRef{
				APIVersion: "kubeflow.org/v1",
				Kind:       "PyTorchJob",
				Namespace:  "ml-train",
				Name:       "llama-pretrain",
				WorldSize:  worldSize,
			},
			DetectWindow:         45 * time.Second,
			ConfirmWindow:        15 * time.Second,
			SMUtilFloorPct:       5.0,
			MinDiskIOBytesPerSec: 10 << 20, // 10 MB/s
			FatalXids:            []int32{79, 94, 95},
			PhaseLeaseTTL: v1alpha1.PhaseLeaseTTL{
				Eval: 3 * time.Minute,
				Ckpt: 5 * time.Minute,
			},
			Remediation: v1alpha1.RemediationSpec{
				TaintKey:       "gpu.infra.io/unhealthy",
				TaintEffect:    "NoSchedule",
				WarmSpareLabel: "pool=gpu-spare",
				DCGMDiagLevel:  3,
				WebhookURL:     "https://bot.internal/v1/incidents",
				CheckpointRoot: ckptRoot,
			},
		},
	}
}

// Milestone 1 Case 1: Rank 7 stalls mid-AllReduce -> isolated & tainted in <60s without blaming innocent waiting ranks.
func TestSingleRankStalledAllReduceIsolatedUnder60s(t *testing.T) {
	ckptRoot, goodDir := seedCheckpoints(t, 8, false)
	now := time.Date(2026, 11, 19, 12, 0, 0, 0, time.UTC)

	samples := make([]telemetry.RankSample, 8)
	for r := 0; r < 8; r++ {
		samples[r] = telemetry.RankSample{
			JobUID:             "job-uid-001",
			NodeName:           fmt.Sprintf("h100-node-%d", r),
			PodName:            fmt.Sprintf("llama-worker-%d", r),
			Rank:               int32(r),
			SMUtilPct:          0.0,
			VRAMUsedMB:         74000,
			LastCompletedSeqID: 4200,
			OpenCollective: &telemetry.CollectiveSpan{
				SeqID:     4201,
				OpName:    "AllReduce",
				StartedAt: now.Add(-46 * time.Second),
				Completed: false,
			},
			ObservedAt: now,
		}
	}
	// Rank 7 never finished seq 4200 and never entered seq 4201 (the true culprit).
	samples[7].LastCompletedSeqID = 4199
	samples[7].OpenCollective = nil

	cluster := &fakeClusterClient{}
	hook := &fakeWebhook{}
	curTime := now
	rec := &TrainingHealthReconciler{
		Telemetry: &fakeTelemetryStore{samples: samples},
		Cluster:   cluster,
		Notifier:  hook,
		Now:       func() time.Time { return curTime },
	}

	policy := basePolicy(ckptRoot, 8)

	// Tick 1 (t=46s): Healthy -> Suspect
	if _, err := rec.Reconcile(context.Background(), policy); err != nil {
		t.Fatal(err)
	}
	if policy.Status.State != v1alpha1.StateSuspect {
		t.Fatalf("expected Suspect, got %s", policy.Status.State)
	}

	// Tick 2 (t=46s): Suspect -> Confirming
	if _, err := rec.Reconcile(context.Background(), policy); err != nil {
		t.Fatal(err)
	}
	if policy.Status.State != v1alpha1.StateConfirming {
		t.Fatalf("expected Confirming, got %s", policy.Status.State)
	}

	// Tick 3 (t=61s, confirmWindow=15s elapsed): Confirming -> Quarantining -> Recovering
	curTime = curTime.Add(15 * time.Second)
	if _, err := rec.Reconcile(context.Background(), policy); err != nil {
		t.Fatal(err)
	}

	if policy.Status.State != v1alpha1.StateRecovering {
		t.Fatalf("expected Recovering, got %s", policy.Status.State)
	}
	if policy.Status.CulpritRank != 7 || policy.Status.CulpritNode != "h100-node-7" {
		t.Fatalf("expected culprit rank 7 (h100-node-7), got rank %d (%s)", policy.Status.CulpritRank, policy.Status.CulpritNode)
	}
	if len(cluster.cordonedNodes) != 1 || cluster.cordonedNodes[0] != "h100-node-7" {
		t.Fatalf("expected h100-node-7 cordoned, got %v", cluster.cordonedNodes)
	}
	if cluster.patchedCkpt != goodDir || cluster.patchedPool != "pool=gpu-spare" {
		t.Fatalf("unexpected PyTorchJob patch: ckpt=%s pool=%s", cluster.patchedCkpt, cluster.patchedPool)
	}
	if len(cluster.diagNodes) != 1 || cluster.diagNodes[0] != "h100-node-7" {
		t.Fatalf("expected DCGM diag on h100-node-7, got %v", cluster.diagNodes)
	}
	if len(hook.events) != 1 || hook.events[0].CulpritRank != 7 {
		t.Fatalf("expected 1 webhook incident for rank 7, got %+v", hook.events)
	}
}

// Milestone 1 Case 2: Slow Lustre checkpoint (SM=0%, openCollective=nil, write_bytes > 10MB/s) -> ZERO false kill.
func TestSlowLustreCheckpointZeroFalsePositive(t *testing.T) {
	ckptRoot, _ := seedCheckpoints(t, 4, false)
	now := time.Date(2026, 11, 19, 12, 0, 0, 0, time.UTC)

	samples := make([]telemetry.RankSample, 4)
	for r := 0; r < 4; r++ {
		samples[r] = telemetry.RankSample{
			JobUID:               "job-uid-001",
			NodeName:             fmt.Sprintf("h100-node-%d", r),
			Rank:                 int32(r),
			SMUtilPct:            0.0,
			VRAMUsedMB:           72000,
			OpenCollective:       nil, // last collective completed cleanly
			LastCompletedSeqID:   8000,
			ProcWriteBytesPerSec: 120 << 20, // 120 MB/s active Lustre flush
			Phase:                "ckpt",
			PhaseLeaseUntil:      now.Add(4 * time.Minute),
			ObservedAt:           now,
		}
	}

	cluster := &fakeClusterClient{}
	rec := &TrainingHealthReconciler{
		Telemetry: &fakeTelemetryStore{samples: samples},
		Cluster:   cluster,
		Now:       func() time.Time { return now },
	}
	policy := basePolicy(ckptRoot, 4)

	for i := 0; i < 4; i++ {
		if _, err := rec.Reconcile(context.Background(), policy); err != nil {
			t.Fatal(err)
		}
	}
	if policy.Status.State != v1alpha1.StateHealthy || len(cluster.cordonedNodes) != 0 {
		t.Fatalf("expected zero false positive during Lustre ckpt flush, state=%s cordoned=%v", policy.Status.State, cluster.cordonedNodes)
	}
}

// Milestone 1 Case 2b: Eval loop with paused collectives but active GPU forward passes -> ZERO false kill.
func TestEvalLoopZeroFalsePositive(t *testing.T) {
	ckptRoot, _ := seedCheckpoints(t, 4, false)
	now := time.Date(2026, 11, 19, 12, 0, 0, 0, time.UTC)

	samples := make([]telemetry.RankSample, 4)
	for r := 0; r < 4; r++ {
		samples[r] = telemetry.RankSample{
			JobUID:             "job-uid-001",
			NodeName:           fmt.Sprintf("h100-node-%d", r),
			Rank:               int32(r),
			SMUtilPct:          62.5, // active forward passes
			VRAMUsedMB:         68000,
			OpenCollective:     nil,
			LastCompletedSeqID: 9000,
			Phase:              "eval",
			PhaseLeaseUntil:    now.Add(2 * time.Minute),
			ObservedAt:         now,
		}
	}

	cluster := &fakeClusterClient{}
	rec := &TrainingHealthReconciler{
		Telemetry: &fakeTelemetryStore{samples: samples},
		Cluster:   cluster,
		Now:       func() time.Time { return now },
	}
	policy := basePolicy(ckptRoot, 4)

	if _, err := rec.Reconcile(context.Background(), policy); err != nil {
		t.Fatal(err)
	}
	if policy.Status.State != v1alpha1.StateHealthy || len(cluster.cordonedNodes) != 0 {
		t.Fatalf("expected zero false positive during eval loop, got state=%s", policy.Status.State)
	}
}

// Milestone 1 Case 3: Wedged checkpoint (write_bytes=0 past 5m lease) + corrupt step 600 -> rolls back cleanly to step 500.
func TestWedgedCheckpointTriggersRollbackToVerifiedNMinus1(t *testing.T) {
	ckptRoot, goodDir := seedCheckpoints(t, 4, true)
	now := time.Date(2026, 11, 19, 12, 0, 0, 0, time.UTC)

	samples := make([]telemetry.RankSample, 4)
	for r := 0; r < 4; r++ {
		samples[r] = telemetry.RankSample{
			JobUID:               "job-uid-001",
			NodeName:             fmt.Sprintf("h100-node-%d", r),
			Rank:                 int32(r),
			SMUtilPct:            0.0,
			VRAMUsedMB:           72000,
			OpenCollective:       nil,
			LastCompletedSeqID:   10000,
			ProcWriteBytesPerSec: 0, // wedged flush
			Phase:                "ckpt",
			PhaseLeaseUntil:      now.Add(-30 * time.Second), // lease expired
			ObservedAt:           now,
		}
	}

	cluster := &fakeClusterClient{}
	curTime := now
	rec := &TrainingHealthReconciler{
		Telemetry: &fakeTelemetryStore{samples: samples},
		Cluster:   cluster,
		Now:       func() time.Time { return curTime },
	}
	policy := basePolicy(ckptRoot, 4)

	_, _ = rec.Reconcile(context.Background(), policy) // Suspect
	_, _ = rec.Reconcile(context.Background(), policy) // Confirming
	curTime = curTime.Add(15 * time.Second)
	if _, err := rec.Reconcile(context.Background(), policy); err != nil {
		t.Fatal(err)
	}

	if policy.Status.State != v1alpha1.StateRecovering {
		t.Fatalf("expected Recovering after wedged checkpoint, got %s", policy.Status.State)
	}
	if cluster.patchedCkpt != goodDir || policy.Status.LastVerifiedCkptStep != 500 || policy.Status.StepsLost != 100 {
		t.Fatalf("expected N-1 rollback to step 500 (%s), got ckpt=%s step=%d lost=%d",
			goodDir, cluster.patchedCkpt, policy.Status.LastVerifiedCkptStep, policy.Status.StepsLost)
	}
}
