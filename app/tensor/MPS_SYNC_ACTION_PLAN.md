# MPS Sync Action Plan

## Context

Recent `conv_medium` profiling shows that the remaining gap is not only GPU
kernel time. Steady-state single-op runs spend roughly 0.38 ms/op in host,
submit, and boundary overhead, while batch profiling shows that delaying sync
across multiple independent convs can cut per-op latency substantially.

Use this plan to turn that result into runtime/API changes without changing
shape routing prematurely.

## Evidence

Primary local targets:

```sh
bazel build -c opt //app/tensor:bench_st_conv_medium_profile
./bazel-bin/app/tensor/bench_st_conv_medium_profile mps_zero_copy_sync
./bazel-bin/app/tensor/bench_st_conv_medium_profile mps_true_async_boundary

bazel build -c opt //app/tensor:bench_st_conv_medium_batch_profile
./bazel-bin/app/tensor/bench_st_conv_medium_batch_profile

bazel build -c opt //app/tensor:bench_st_pipeline
./bazel-bin/app/tensor/bench_st_pipeline
```

Key interpretation from the batch run:

- `batch_ops=1`: delaying sync is not useful by itself.
- `batch_ops>=2`: `batched_sync_end` reduces `ms_per_op` and `sync_ms_per_op`.
- Valid primary metrics are `ms_per_op`, `sync_ms_per_op`, `mps_hit`, and
  fallback counters.
- `cpu_overhead_ms` is diagnostic only in batch mode because `gpu_avg_ms` and
  `ms_per_op` are not guaranteed to share the same effective aggregation
  boundary.

## Runtime Rules

1. Keep MPS outputs GPU-resident by default.

   A conv output backed by a Metal buffer should stay in that buffer until a
   true CPU boundary is reached. Avoid CPU materialization for intermediate
   tensors.

2. Sync only at true boundaries.

   Valid boundaries are explicit user sync, CPU value access, correctness
   comparison, timing end, object destruction, or a backend transition that
   must read CPU memory.

3. Do not overwrite a pending output without draining it.

   Reusing the same output tensor for another async write must wait for the
   previous command buffer first. This preserves correctness but also explains
   why repeated writes to one output do not represent ideal batching.

4. Routing should consider downstream residency.

   `conv_medium` remains an MPS candidate, especially when later work can stay
   on GPU. Isolated single convs still pay a large boundary component.

## Priority Changes

### P1: Audit implicit CPU boundaries

Find production paths that read `tensor->values` after an async MPS op without
an explicit boundary decision.

Checks:

```sh
rg -n "st_tensor_sync|->values|readBytes|conv_readbytes" app/tensor/src app/tensor/include
```

Expected outcome:

- Intentional sync points have a reason.
- CPU-only ops either sync explicitly before reading or are not called with
  pending GPU tensors.
- `conv_readbytes` remains zero in inference fast paths.

### P2: Prefer end-of-pipeline sync in callers

Callers that execute multiple GPU-capable ops should synchronize once at the
end instead of after every op.

Verification:

- `bench_st_conv_medium_batch_profile`: `batched_sync_end` wins for
  `batch_ops>=2`.
- `bench_st_pipeline`: `boundary_sync_only` should beat `sync_each_iter` for
  MPS-backed medium/large cases.

### P3: Add pipeline-like coverage before adding routing thresholds

Before changing MPS/GEMM thresholds for `conv_medium`, verify the workload
class:

- isolated single conv
- repeated independent convs
- fused `conv+bn`
- fused `conv+bn+pool`

Only encode a threshold if the benchmark result includes the target, date,
hardware, and shape class.

## Do Not Do Yet

- Do not route `conv_medium` away from MPS solely because isolated single-op
  latency is worse than PyTorch/MLX.
- Do not assume `mps_true_async_boundary` is universally faster than
  `mps_zero_copy_sync`.
- Do not treat negative `cpu_overhead_ms` in batch mode as a real negative CPU
  cost.
- Do not introduce broad graph/session abstractions until the existing explicit
  sync policy is exhausted.

