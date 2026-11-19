package telemetry

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// CollectiveSpan mirrors a single NCCL collective entry from PyTorch Flight Recorder.
type CollectiveSpan struct {
	SeqID     uint64    `json:"seq_id"`
	PGName    string    `json:"pg_name"`
	OpName    string    `json:"op_name"`
	StartedAt time.Time `json:"started_at"`
	Completed bool      `json:"completed"`
}

// PhaseLease is written atomically by the training process to /dev/shm/phase.json.
type PhaseLease struct {
	Phase     string    `json:"phase"` // "train" | "eval" | "ckpt"
	ExpiresAt time.Time `json:"expires_at"`
}

// RankSample is emitted every 10s by the node-agent DaemonSet for each local GPU rank.
type RankSample struct {
	JobUID               string          `json:"job_uid"`
	NodeName             string          `json:"node_name"`
	PodName              string          `json:"pod_name"`
	Rank                 int32           `json:"rank"`
	SMUtilPct            float32         `json:"sm_util_pct"`
	VRAMUsedMB           uint64          `json:"vram_used_mb"`
	OpenCollective       *CollectiveSpan `json:"open_collective,omitempty"`
	LastCompletedSeqID   uint64          `json:"last_completed_seq_id"`
	ProcWriteBytesPerSec uint64          `json:"proc_write_bytes_per_sec"`
	IBErrDelta           uint64          `json:"ib_err_delta"`
	FatalXids            []int32         `json:"fatal_xids,omitempty"`
	Phase                string          `json:"phase"`
	PhaseLeaseUntil      time.Time       `json:"phase_lease_until"`
	ObservedAt           time.Time       `json:"observed_at"`
}

// CollectorMode indicates whether CO-RE eBPF tracing is active or degraded to sysfs/procfs polling.
type CollectorMode string

const (
	ModeCOREEBPF    CollectorMode = "co-re-ebpf"
	ModeSysfsProcfs CollectorMode = "sysfs-procfs-fallback"
)

// DetectCollectorMode runs a BTF preflight check; if /sys/kernel/btf/vmlinux is absent,
// the DaemonSet gracefully degrades to /proc/<pid>/io + /sys/class/infiniband + NVML polling.
func DetectCollectorMode(btfPath string) CollectorMode {
	if btfPath == "" {
		btfPath = "/sys/kernel/btf/vmlinux"
	}
	if fi, err := os.Stat(btfPath); err == nil && !fi.IsDir() && fi.Size() > 0 {
		return ModeCOREEBPF
	}
	return ModeSysfsProcfs
}

// ReadProcWriteBytes parses write_bytes from Linux /proc/<pid>/io.
func ReadProcWriteBytes(procIOPath string) (uint64, error) {
	f, err := os.Open(procIOPath)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if strings.HasPrefix(line, "write_bytes:") {
			fields := strings.Fields(line)
			if len(fields) == 2 {
				return strconv.ParseUint(fields[1], 10, 64)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return 0, err
	}
	return 0, fmt.Errorf("write_bytes not found in %s", procIOPath)
}

// ReadIBPortErrors aggregates link_downed and port_rcv_errors from /sys/class/infiniband.
func ReadIBPortErrors(ibRoot string) (uint64, error) {
	if ibRoot == "" {
		ibRoot = "/sys/class/infiniband"
	}
	var total uint64
	counters := []string{"port_rcv_errors", "link_downed", "port_xmit_discards"}
	matches, err := filepath.Glob(filepath.Join(ibRoot, "*", "ports", "*", "counters"))
	if err != nil {
		return 0, err
	}
	for _, dir := range matches {
		for _, name := range counters {
			raw, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				continue
			}
			v, err := strconv.ParseUint(strings.TrimSpace(string(raw)), 10, 64)
			if err == nil {
				total += v
			}
		}
	}
	return total, nil
}

// ReadPhaseLease reads the optional bounded /dev/shm/phase.json lease file.
func ReadPhaseLease(path string) (PhaseLease, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return PhaseLease{}, err
	}
	var lease PhaseLease
	if err := json.Unmarshal(raw, &lease); err != nil {
		return PhaseLease{}, err
	}
	return lease, nil
}
