# nccl-hang-operator

Kubernetes failure-detection and auto-recovery operator for distributed GPU training (`PyTorchJob` / `JobSet`). Cuts silent NCCL collective hang detection from the default 10–30 minute watchdog timeout down to **<60 seconds**, isolates the exact culprit rank/node, verifies atomic sharded checkpoints with automatic `N-1` rollback, and triggers automated RCA + PR generation via webhook.

## Why

Default NCCL watchdogs (`TORCH_NCCL_ASYNC_ERROR_HANDLING` / `NCCL_TIMEOUT`) sit idle for 10–30 minutes burning GPU-hours before crashing the entire job—without identifying which node actually caused the stall. Naive "SM=0% + no comms" detectors also false-kill healthy jobs during evaluation loops or slow Lustre/NFS checkpoint syncs.

## Architecture

1. **Userspace PyTorch Flight Recorder + Zero-Overhead NIC ASIC Counters + Slow-Path CO-RE eBPF (`cmd/node-agent`, `internal/telemetry`)**
   - Flight Recorder stays strictly in userspace (`TORCH_NCCL_TRACE_BUFFER_SIZE=2000` dumped via `/dev/shm` shared `emptyDir`, zero kernel risk).
   - **Zero-Overhead 400Gbps RDMA Telemetry**: Because NCCL runs userspace kernel-bypass RDMA (`libibverbs` / `mlx5` doorbell rings + GPUDirect), collective WQEs never hit kernel `ib_post_send` and must never be trapped on the hot path. Instead, `node-agent` polls ConnectX NIC ASIC hardware counters every 1–5s via `rdma` netlink (`RDMA_NLDEV_CMD_STAT_GET`) / sysfs (`port_xmit_data`, `port_rcv_data`, and `hw_counters` such as `local_ack_timeout_err`, `packet_seq_err`, `out_of_sequence`) at line rate with **0% datapath overhead**.
   - **Slow/Error-Path CO-RE eBPF Only**: Least-privilege CO-RE eBPF (`CAP_BPF` + `CAP_PERFMON`, no `--privileged`) attaches strictly off the 400Gbps hot path — `ib_dispatch_event` (QP fatal / port-down faults only), `tcp_retransmit_skb` (OOB bootstrap stalls), and `vfs_write` filtered by training PID into a `BPF_MAP_TYPE_PERCPU_ARRAY` (no per-syscall ringbuf spam) for checkpoint flush detection — with an automatic BTF preflight check (`DetectCollectorMode`) that gracefully degrades to `/proc/<pid>/io` + `/sys/class/infiniband` + NVML polling on older kernels.
2. **False-Positive-Free Hang Detection & Culprit Isolation (`internal/controller`)**
   - **Open vs. Completed Collectives + RDMA Progress Gating**: Distinguishes deadlocked open NCCL collectives (`Completed == false`, zero RDMA byte progress) from legitimate slow cross-switch InfiniBand/RoCE `AllGather` stragglers that are still actively transferring bytes (`RDMAProgressBytesPerSec >= 1MB/s`, zero IB/HW errors).
   - **Eval & Checkpoint Backoff**: Suppresses false kills when GPU forward passes remain active (`SMUtilPct >= 5%`), storage write throughput is climbing (`ProcWriteBytesPerSec >= 10MB/s`), or a bounded `/dev/shm/phase.json` lease (`eval: 3m`, `ckpt: 5m`) is active.
   - **Two-Stage FSM**: `Healthy -> Suspect -> Confirming -> Quarantining -> Recovering`.
   - **Culprit Ranking**: Prioritizes fatal NVML Xids (`79, 94, 95`) and InfiniBand/ConnectX `hw_counters` error deltas, then diffs Flight Recorder `LastCompletedSeqID` across ranks (`argmin`) to isolate the lagging culprit instead of innocent waiting ranks.
3. **Atomic Sharded Checkpoint Verifier, `N-1` Rollback & Async Devin Bot RCA (`internal/checkpoint`, `internal/controller`)**
   - Rejects directories with leftover `.tmp`/`.incomplete` artifacts.
   - Validates `world_size` shard completeness, step parity across model weights / optimizer / dataloader, SHA256 checksums, and `.safetensors` 8-byte JSON header offset bounds (catching truncated writes without loading 100GB tensors into RAM).
   - Automatically rolls back to the newest verified `N-1` step, patches `PyTorchJob` with an **immutable pinned checkpoint path** (never `latest`), and dispatches an `IncidentPayload` webhook to **Devin Bot** for async root-cause analysis (opening training-code PRs for rank `seq_id` mismatches or node RCA/RMA tickets for Xid/fabric faults).

## Project Layout

```text
api/v1alpha1/traininghealthpolicy_types.go      # TrainingHealthPolicy CRD Go types
config/crd/nccl.infra.io_traininghealthpolicies.yaml
config/samples/nccl_v1alpha1_traininghealthpolicy.yaml
config/deploy/daemonset-agent.yaml              # Least-privilege CAP_BPF + CAP_PERFMON DaemonSet
internal/telemetry/sample.go                    # Flight Recorder / procfs / IB / BTF preflight
internal/checkpoint/verifier.go                 # DCP + safetensors header & N-1 rollback verifier
internal/controller/reconciler.go               # 5-state FSM reconciler + Devin Bot webhook
cmd/controller/main.go                          # Controller entrypoint
cmd/node-agent/main.go                          # Node-agent DaemonSet entrypoint
```

## Quick Verification (Milestone 1 Deterministic Harness)

```bash
make vet
make build
make test
```

Deterministic fault-injection tests cover:
- **Single-rank stalled `AllReduce`** (rank 7 lags on `seq_id`) -> isolated, cordoned, tainted, DCGM Level 3 scheduled, and incident webhook dispatched in `<60s`.
- **Slow Lustre checkpoint flush** (`SM=0%`, `openCollective=nil`, `120 MB/s` disk write) -> zero false kills.
- **Evaluation loop** (paused training collectives, `62.5%` SM forward passes) -> zero false kills.
- **Wedged checkpoint past lease + truncated `.safetensors` shard on step `K`** -> catches wedged flush and automatically rolls `PyTorchJob` back to verified step `K-1`.
- **Slow cross-switch InfiniBand `AllGather` straggler** (`OpenCollective=AllGather` past `DetectWindow + ConfirmWindow`, `port_xmit_data` moving at `450 MB/s`, zero IB/HW errors) -> zero false kills.
- **ConnectX ASIC `hw_counters` fault** (`local_ack_timeout_err` / `packet_seq_err` burst) -> isolates the faulty node immediately and dispatches the Devin Bot incident webhook.
