package checkpoint

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Manifest models the atomic-renamed commit manifest written per checkpoint step
// (compatible with PyTorch DCP .metadata.json / Megatron-Core sharded layouts).
type Manifest struct {
	Step           int64             `json:"step"`
	WorldSize      int               `json:"world_size"`
	OptimizerStep  int64             `json:"optimizer_step"`
	DataloaderStep int64             `json:"dataloader_step"`
	Shards         map[string]string `json:"shards"` // filename -> sha256 (or xxHash64/ETag)
}

// VerifiedCheckpoint is an immutable, verified checkpoint target for PyTorchJob resumption.
type VerifiedCheckpoint struct {
	Path      string
	Step      int64
	StepsLost int64
}

// SelectVerified walks checkpoint directories newest-first (rolling N, N-1, N-2 window),
// validates completeness, structural headers, step parity, and checksums, and returns
// an immutable pinned directory path (never "latest").
func SelectVerified(rootDir string, worldSize int) (*VerifiedCheckpoint, error) {
	entries, err := os.ReadDir(rootDir)
	if err != nil {
		return nil, fmt.Errorf("read checkpoint root: %w", err)
	}

	var dirs []string
	for _, e := range entries {
		if e.IsDir() && strings.HasPrefix(e.Name(), "step-") {
			dirs = append(dirs, filepath.Join(rootDir, e.Name()))
		}
	}
	if len(dirs) == 0 {
		return nil, fmt.Errorf("no step-* checkpoint directories under %s", rootDir)
	}

	// Sort descending by step number so newest is checked first, rolling back to N-1 on failure.
	sort.Slice(dirs, func(i, j int) bool {
		return parseStepDir(dirs[i]) > parseStepDir(dirs[j])
	})

	newestStep := parseStepDir(dirs[0])
	var lastErr error
	for _, dir := range dirs {
		m, err := VerifyDir(dir, worldSize)
		if err != nil {
			lastErr = err
			continue
		}
		stepsLost := newestStep - m.Step
		if stepsLost < 0 {
			stepsLost = 0
		}
		return &VerifiedCheckpoint{
			Path:      dir,
			Step:      m.Step,
			StepsLost: stepsLost,
		}, nil
	}
	return nil, fmt.Errorf("all %d checkpoint candidates failed verification (last error: %w)", len(dirs), lastErr)
}

// VerifyDir validates a single sharded checkpoint directory without loading full tensor weights into RAM.
func VerifyDir(dir string, worldSize int) (*Manifest, error) {
	// Reject directories containing leftover temporary files from interrupted async flushes.
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") || strings.HasSuffix(e.Name(), ".incomplete") {
			return nil, fmt.Errorf("incomplete flush artifact %q in %s", e.Name(), dir)
		}
	}

	raw, err := os.ReadFile(filepath.Join(dir, ".metadata.json"))
	if err != nil {
		return nil, fmt.Errorf("missing atomic commit manifest in %s: %w", dir, err)
	}
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("corrupt manifest JSON in %s: %w", dir, err)
	}

	if m.WorldSize != worldSize || len(m.Shards) != worldSize {
		return nil, fmt.Errorf("shard count mismatch in %s: manifest=%d/%d expected=%d", dir, len(m.Shards), m.WorldSize, worldSize)
	}
	if m.Step <= 0 || m.Step != m.OptimizerStep || m.Step != m.DataloaderStep {
		return nil, fmt.Errorf("step parity mismatch in %s: weights=%d opt=%d loader=%d", dir, m.Step, m.OptimizerStep, m.DataloaderStep)
	}

	for shardName, expectedHash := range m.Shards {
		shardPath := filepath.Join(dir, shardName)
		fi, err := os.Stat(shardPath)
		if err != nil {
			return nil, fmt.Errorf("missing shard %s: %w", shardName, err)
		}
		if fi.Size() == 0 {
			return nil, fmt.Errorf("empty shard %s", shardName)
		}

		// Fast structural header bounds validation for .safetensors shards.
		if strings.HasSuffix(shardName, ".safetensors") {
			if err := verifySafetensorsHeader(shardPath, fi.Size()); err != nil {
				return nil, fmt.Errorf("corrupt safetensors header in %s: %w", shardName, err)
			}
		}

		if expectedHash != "" {
			gotHash, err := fileSHA256(shardPath)
			if err != nil {
				return nil, err
			}
			if !strings.EqualFold(gotHash, expectedHash) {
				return nil, fmt.Errorf("checksum mismatch for %s: got=%s want=%s", shardName, gotHash, expectedHash)
			}
		}
	}

	return &m, nil
}

type safetensorsEntry struct {
	DataOffsets [2]int64 `json:"data_offsets"`
}

// verifySafetensorsHeader reads only the 8-byte length prefix + JSON header and validates
// that every tensor's byte offset fits strictly within the file size (catching truncated writes).
func verifySafetensorsHeader(path string, fileSize int64) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return fmt.Errorf("read header length: %w", err)
	}
	if headerLen == 0 || int64(8+headerLen) > fileSize || headerLen > 64<<20 {
		return fmt.Errorf("invalid header length %d for file size %d", headerLen, fileSize)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return fmt.Errorf("truncated header: %w", err)
	}

	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawMap); err != nil {
		return fmt.Errorf("invalid header JSON: %w", err)
	}

	dataSize := fileSize - int64(8+headerLen)
	for k, v := range rawMap {
		if k == "__metadata__" {
			continue
		}
		var meta safetensorsEntry
		if err := json.Unmarshal(v, &meta); err != nil {
			return fmt.Errorf("invalid tensor entry %s: %w", k, err)
		}
		start, end := meta.DataOffsets[0], meta.DataOffsets[1]
		if start < 0 || end < start || end > dataSize {
			return fmt.Errorf("tensor %s offsets [%d,%d] exceed payload size %d", k, start, end, dataSize)
		}
	}
	return nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func parseStepDir(dir string) int64 {
	base := filepath.Base(dir)
	var step int64
	_, _ = fmt.Sscanf(base, "step-%d", &step)
	return step
}
