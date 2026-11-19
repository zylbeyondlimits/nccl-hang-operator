package telemetry

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDetectCollectorModeFallback(t *testing.T) {
	if mode := DetectCollectorMode(filepath.Join(t.TempDir(), "missing-btf")); mode != ModeSysfsProcfs {
		t.Fatalf("expected sysfs/procfs fallback when BTF is missing, got %s", mode)
	}

	btf := filepath.Join(t.TempDir(), "vmlinux")
	if err := os.WriteFile(btf, []byte("btf-stub"), 0o644); err != nil {
		t.Fatal(err)
	}
	if mode := DetectCollectorMode(btf); mode != ModeCOREEBPF {
		t.Fatalf("expected CO-RE eBPF mode when BTF exists, got %s", mode)
	}
}

func TestReadProcWriteBytesAndIBCounters(t *testing.T) {
	dir := t.TempDir()
	procIO := filepath.Join(dir, "io")
	if err := os.WriteFile(procIO, []byte("rchar: 123\nwchar: 456\nwrite_bytes: 52428800\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	wb, err := ReadProcWriteBytes(procIO)
	if err != nil || wb != 52428800 {
		t.Fatalf("expected write_bytes=52428800, got %d err=%v", wb, err)
	}

	ibCounters := filepath.Join(dir, "mlx5_0", "ports", "1", "counters")
	if err := os.MkdirAll(ibCounters, 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(ibCounters, "port_rcv_errors"), []byte("3\n"), 0o644)
	_ = os.WriteFile(filepath.Join(ibCounters, "link_downed"), []byte("1\n"), 0o644)
	_ = os.WriteFile(filepath.Join(ibCounters, "port_xmit_data"), []byte("1000\n"), 0o644)
	_ = os.WriteFile(filepath.Join(ibCounters, "port_rcv_data"), []byte("2500\n"), 0o644)

	ibHWCounters := filepath.Join(dir, "mlx5_0", "ports", "1", "hw_counters")
	if err := os.MkdirAll(ibHWCounters, 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(ibHWCounters, "local_ack_timeout_err"), []byte("2\n"), 0o644)
	_ = os.WriteFile(filepath.Join(ibHWCounters, "packet_seq_err"), []byte("5\n"), 0o644)

	errs, err := ReadIBPortErrors(dir)
	if err != nil || errs != 4 {
		t.Fatalf("expected total IB errors=4, got %d err=%v", errs, err)
	}

	// IBTA port_xmit_data + port_rcv_data are reported in 4-byte octets: (1000 + 2500) * 4 = 14000 bytes.
	dataBytes, err := ReadIBDataBytes(dir)
	if err != nil || dataBytes != 14000 {
		t.Fatalf("expected total IB data bytes=14000, got %d err=%v", dataBytes, err)
	}

	hwErrs, err := ReadIBHWErrors(dir)
	if err != nil || hwErrs != 7 {
		t.Fatalf("expected total ConnectX HW errors=7, got %d err=%v", hwErrs, err)
	}

	leasePath := filepath.Join(dir, "phase.json")
	exp := time.Date(2026, 11, 19, 12, 5, 0, 0, time.UTC)
	_ = os.WriteFile(leasePath, []byte(`{"phase":"ckpt","expires_at":"2026-11-19T12:05:00Z"}`), 0o644)
	lease, err := ReadPhaseLease(leasePath)
	if err != nil || lease.Phase != "ckpt" || !lease.ExpiresAt.Equal(exp) {
		t.Fatalf("unexpected phase lease: %+v err=%v", lease, err)
	}
}
