package v1alpha1

import "time"

// HealthState represents the finite state machine phase of a monitored training job.
type HealthState string

const (
	StateHealthy      HealthState = "Healthy"
	StateSuspect      HealthState = "Suspect"
	StateConfirming   HealthState = "Confirming"
	StateQuarantining HealthState = "Quarantining"
	StateRecovering   HealthState = "Recovering"
)

// TargetRef identifies the distributed training workload (PyTorchJob or JobSet).
type TargetRef struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Namespace  string `json:"namespace"`
	Name       string `json:"name"`
	WorldSize  int    `json:"worldSize"`
}

// PhaseLeaseTTL defines bounded backoff windows during legitimate non-collective phases.
type PhaseLeaseTTL struct {
	Eval time.Duration `json:"eval"`
	Ckpt time.Duration `json:"ckpt"`
}

// RemediationSpec configures automated node quarantine, diagnostics, and webhook reporting.
type RemediationSpec struct {
	TaintKey       string `json:"taintKey"`
	TaintEffect    string `json:"taintEffect"`
	WarmSpareLabel string `json:"warmSpareLabel"`
	DCGMDiagLevel  int    `json:"dcgmDiagLevel"`
	WebhookURL     string `json:"webhookURL,omitempty"`
	CheckpointRoot string `json:"checkpointRoot"`
}

// TrainingHealthPolicySpec defines the desired detection and auto-recovery policy.
type TrainingHealthPolicySpec struct {
	TargetRef            TargetRef       `json:"targetRef"`
	DetectWindow         time.Duration   `json:"detectWindow"`
	ConfirmWindow        time.Duration   `json:"confirmWindow"`
	SMUtilFloorPct       float32         `json:"smUtilFloorPct"`
	MinDiskIOBytesPerSec uint64          `json:"minDiskIoBytesPerSec"`
	MinRDMABytesPerSec   uint64          `json:"minRdmaBytesPerSec"`
	FatalXids            []int32         `json:"fatalXids"`
	PhaseLeaseTTL        PhaseLeaseTTL   `json:"phaseLeaseTTL"`
	Remediation          RemediationSpec `json:"remediation"`
}

// TrainingHealthPolicyStatus observes the live controller state for the workload.
type TrainingHealthPolicyStatus struct {
	State                HealthState `json:"state"`
	SuspectSince         *time.Time  `json:"suspectSince,omitempty"`
	ConfirmingSince      *time.Time  `json:"confirmingSince,omitempty"`
	CulpritNode          string      `json:"culpritNode,omitempty"`
	CulpritRank          int32       `json:"culpritRank,omitempty"`
	CulpritReason        string      `json:"culpritReason,omitempty"`
	LastVerifiedCkptPath string      `json:"lastVerifiedCkptPath,omitempty"`
	LastVerifiedCkptStep int64       `json:"lastVerifiedCkptStep,omitempty"`
	StepsLost            int64       `json:"stepsLost,omitempty"`
	GPUHoursSaved        float64     `json:"gpuHoursSaved,omitempty"`
	LastTransitionTime   time.Time   `json:"lastTransitionTime"`
}

// TrainingHealthPolicy is the schema for the nccl.infra.io/v1alpha1 CRD.
type TrainingHealthPolicy struct {
	APIVersion string                     `json:"apiVersion"`
	Kind       string                     `json:"kind"`
	Namespace  string                     `json:"namespace"`
	Name       string                     `json:"name"`
	UID        string                     `json:"uid"`
	Spec       TrainingHealthPolicySpec   `json:"spec"`
	Status     TrainingHealthPolicyStatus `json:"status"`
}
