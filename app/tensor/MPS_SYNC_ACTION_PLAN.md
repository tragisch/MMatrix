# MPS Sync Action Plan

## Status

**P1: CPU-boundary audit** ✅ COMPLETED (2026-05-03)
- `st_gpu_guard` hook: always-active, logs `[ST_GPU_GUARD]` and increments counter
- 7 unit tests: 5 synthetic (Sentinel-based) + 1 real MPS async P2 test (skips gracefully if fastpath unavailable)
- Guards applied to high-risk CPU APIs: `st_get`, `st_set`, `st_clone`, `st_to_f32`, `st_to_bf16`, `st_inplace_*`, `st_sum_axes`, `st_pad_nchw`
- Outcome: 0 violations in current unit/integration test corpus
- Next: P2 caller-side sync bundling before routing changes

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

Current callsite classification:

| Bucket                    | Callsites                                                                                                    | Interpretation                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Intentional               | `st_tensor_sync`, `st_buffer_wait_gpu`, `st_buffer_release`, MPS `readBytes` fallback/training paths         | These are explicit boundaries or safety drains.                                                                                               |
| Harmless in current tests | Unit-test and layout-benchmark calls to `st_get`, `st_clone`, `st_to_f32`, `st_sum_axes`, `st_pad_nchw`      | CPU tensors or intentional materialization benchmarks.                                                                                        |
| Watch                     | BF16 promotion in `st_conv.c`, `st_batchnorm.c`, `st_pool.c`, and `st.c`                                     | These call `st_to_f32` / `st_clone` and therefore read CPU memory. Safe for CPU tensors; expensive or wrong if called on pending GPU tensors. |
| Watch                     | CPU fallback path in fused `conv+bn` / `conv+bn+pool`                                                        | If MPS fused execution fails after producing or receiving a GPU-resident intermediate, fallback work must not read stale `->values`.          |
| High-risk API contract    | `st_get`, `st_set`, `st_clone`, `st_to_f32`, `st_to_bf16`, `st_sum_axes`, `st_pad_nchw`, and CPU inplace ops | These are CPU APIs. Callers must treat them as boundaries when the tensor may be GPU-pending.                                                 |

Near-term rule:

- Before any CPU API reads or writes `tensor->values`, either the caller must
  have already synchronized explicitly or the API must synchronize internally.
- Do not add internal sync blindly to every scalar/helper API yet; first measure
  whether this hides expensive CPU materialization in hot paths.
- Prefer documenting and testing the boundary contract before changing default
  behavior.

Implemented guard:

- `st_gpu_guard` logs `[ST_GPU_GUARD]` and increments a counter when selected
  CPU APIs are called on a tensor with `_async_cmd_buf != NULL`.
- The guard is diagnostic only. It never calls `st_tensor_sync()` and does not
  change runtime semantics.
- Local verification: only synthetic assertions in `test_st_gpu_guard` trigger
  the hook. The current productive tensor test corpus did not trigger guard
  violations.
- Interpretation: the known risk is currently not covered by regular unit
  tests; it is expected to show up in runtime scenarios such as `MPS conv ->
  st_get` or `MPS conv -> st_clone` without explicit sync.

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
