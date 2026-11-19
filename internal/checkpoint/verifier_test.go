package checkpoint

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func writeValidSafetensors(t *testing.T, path string, payloadBytes int) string {
	t.Helper()
	header := fmt.Sprintf(`{"weight":{"dtype":"BF16","shape":[%d],"data_offsets":[0,%d]}}`, payloadBytes/2, payloadBytes)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := binary.Write(f, binary.LittleEndian, uint64(len(header))); err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(header); err != nil {
		t.Fatal(err)
	}
	payload := make([]byte, payloadBytes)
	for i := range payload {
		payload[i] = byte(i % 251)
	}
	if _, err := f.Write(payload); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func writeStepDir(t *testing.T, root string, step int64, worldSize int, corruptMode string) string {
	t.Helper()
	dir := filepath.Join(root, fmt.Sprintf("step-%d", step))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	shards := make(map[string]string, worldSize)
	for r := 0; r < worldSize; r++ {
		name := fmt.Sprintf("rank-%d.safetensors", r)
		p := filepath.Join(dir, name)
		hash := writeValidSafetensors(t, p, 64)
		if corruptMode == "truncated_safetensors" && r == worldSize-1 {
			// Truncate payload so header data_offsets exceed actual file length.
			raw, _ := os.ReadFile(p)
			_ = os.WriteFile(p, raw[:len(raw)-20], 0o644)
			sum := sha256.Sum256(raw[:len(raw)-20])
			hash = hex.EncodeToString(sum[:])
		}
		shards[name] = hash
	}

	if corruptMode == "leftover_tmp" {
		_ = os.WriteFile(filepath.Join(dir, "rank-0.safetensors.tmp"), []byte("partial"), 0o644)
	}

	m := Manifest{
		Step:           step,
		WorldSize:      worldSize,
		OptimizerStep:  step,
		DataloaderStep: step,
		Shards:         shards,
	}
	if corruptMode == "step_parity_mismatch" {
		m.OptimizerStep = step - 1
	}

	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".metadata.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestSelectVerifiedRollbackToNMinus1OnTruncatedShard(t *testing.T) {
	root := t.TempDir()
	goodDir := writeStepDir(t, root, 1000, 4, "")
	_ = writeStepDir(t, root, 1100, 4, "truncated_safetensors")

	got, err := SelectVerified(root, 4)
	if err != nil {
		t.Fatalf("expected N-1 fallback to succeed, got err: %v", err)
	}
	if got.Path != goodDir || got.Step != 1000 || got.StepsLost != 100 {
		t.Fatalf("unexpected rollback result: %+v", got)
	}
}

func TestVerifyDirRejectsLeftoverTmpAndParityMismatch(t *testing.T) {
	root := t.TempDir()
	tmpDir := writeStepDir(t, root, 200, 2, "leftover_tmp")
	if _, err := VerifyDir(tmpDir, 2); err == nil {
		t.Fatalf("expected leftover .tmp file to fail verification")
	}

	parityDir := writeStepDir(t, root, 300, 2, "step_parity_mismatch")
	if _, err := VerifyDir(parityDir, 2); err == nil {
		t.Fatalf("expected step parity mismatch to fail verification")
	}
}
