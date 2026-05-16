# MPS Sync Action Plan

## Status

**P1: CPU-boundary audit** ✅ COMPLETED (2026-05-03)
- `st_gpu_guard` hook: always-active, logs `[ST_GPU_GUARD]` and increments counter
- 7 unit tests: clean tensor baseline, 5 synthetic/Sentinel guard checks, and
  1 real MPS async `conv -> st_get` test (skips gracefully if fastpath unavailable)
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
- Local verification: synthetic assertions trigger the hook, and the dedicated
  real async `MPS conv -> st_get` test triggers it when the fastpath is
  available. The broader productive tensor test corpus did not trigger
  unexpected guard violations.
- Interpretation: the known risk is now covered by one real pending-GPU path,
  but not by every CPU API. Extend real-path coverage incrementally before
  broadening runtime behavior.

### P2: Prefer end-of-pipeline sync in callers

Callers that execute multiple GPU-capable ops should synchronize once at the
end instead of after every op.

Verification:

- `bench_st_conv_medium_batch_profile`: `batched_sync_end` wins for
  `batch_ops>=2`.
- `bench_st_pipeline`: compare `boundary_sync_only` against `sync_each_iter`
  for MPS-backed medium/large cases; do not assume the boundary-only variant
  wins for every fused pipeline shape.

Current local snapshot (2026-05-14, macOS/Apple Silicon):

- `bench_st_conv_medium_batch_profile` confirms P2 direction for independent
  conv batches: `batched_sync_end` beats `serial_sync_each` for `batch_ops>=2`
  (e.g. batch 8: `0.336 ms/op` vs `0.677 ms/op`).
- `bench_st_pipeline` was updated to force MPS request (`ST_CONV_BACKEND_MPS`)
  and explicit async mode (`st_backend_set_conv_mps_async(true)`) for cleaner
  boundary-policy comparison.
- Fused pipeline result is currently mixed:
  - `conv+bn`: large favors boundary sync (`27.57` vs `38.85 ms/iter`),
    medium slightly favors sync-each (`2.22` vs `2.40 ms/iter`).
  - `conv+bn+pool`: medium/large currently favor sync-each in this run.

Interpretation:

- P2 is validated for independent conv batching.
- For fused pipelines, we should not enforce a blanket routing/sync policy yet;
  we need one more pass with stable repetition and variance capture before
  changing defaults.

### P3: Add pipeline-like coverage before adding routing thresholds

Before changing MPS/GEMM thresholds for `conv_medium`, verify the workload
class:

- isolated single conv
- repeated independent convs
- fused `conv+bn`
- fused `conv+bn+pool`

Only encode a threshold if the benchmark result includes the target, date,
hardware, and shape class.

### P4: Cross-framework parity track (MLX / PyTorch as success metric)

Goal:

- Optimize tensor op runtime against real external baselines, not only internal
  A/B comparisons.
- Primary external references: MLX and PyTorch (MPS backend), same machine,
  same shape, same dtype, synchronized timing boundaries.

Execution target:

- Use `tools/bench_conv_cross_framework.py` as the baseline harness.
- Keep focus shape set aligned with internal routing decisions:
  `conv_medium`, `conv_large`, `pw_medium`.
- Extend the harness later for `conv+bn` and `conv+bn+pool` once the
  single-op parity line is stable.

Measurement rules (must hold for every published run):

1. Same hardware snapshot in output (`platform`, CPU model, memory).
2. Warmup and iteration counts documented; report median over repeated runs.
3. Timing includes explicit sync/materialization boundaries for all frameworks.
4. Compare with `ms/iter` and `GMAC/s`; reject results with mismatched shape
   semantics or lazy-eval leakage.

Acceptance bands for current optimization phase (conv focus):

- `conv_large` (MPS candidate): MMatrix MPS path within **1.15x** of the faster
  of MLX/PyTorch.
- `conv_medium` (borderline): MMatrix best path within **1.25x** of the faster
  of MLX/PyTorch.
- `pw_medium` (1x1): stay on GEMM fast path; within **1.20x** of the faster
  external baseline.

If a case misses the band:

- classify as one of: dispatch threshold miss, sync-boundary overhead,
  host/device transfer overhead, kernel implementation gap.
- open a targeted optimization item with measured deltas before changing global
  thresholds.

Near-term next step:

- Run cross-framework benchmark with fixed repeats and archive result snapshot
  (date + machine + commit) before further routing changes.
- Use that snapshot as the regression gate for all upcoming MPS sync and
  dispatch changes.

Readiness gate (observed on 2026-05-16, local machine):

- Python deps missing for external baselines: `torch`, `mlx`.
- Local Bazel build of `//app/tensor:bench_st_ab_conv` blocked by
  Xcode-version mismatch in toolchain resolution (`26.4.0.17E192` expected vs
  local `26.5`).

Unblock order:

1. Fix Bazel/Xcode toolchain resolution and rebuild `bench_st_ab_conv`.
2. Install/verify Python baseline deps (`torch`, `mlx`) for the active
   interpreter.
3. Re-run `tools/bench_conv_cross_framework.py` and archive first parity
   snapshot.

First parity snapshot (2026-05-16, local Apple Silicon, repeats=3):

| Case          | c_gemm ms | c_mps_zero_copy_sync ms | c_mps_true_async_boundary ms | pytorch_mps ms | mlx ms | Winner             |
| ------------- | --------- | ----------------------- | ---------------------------- | -------------- | ------ | ------------------ |
| conv_medium   | 6.42      | 1.31                    | 1.86                         | 0.71           | 0.64   | MLX                |
| conv_large    | 169.94    | 7.09                    | 10.61                        | 8.70           | 10.52  | MMatrix (MPS sync) |
| pw_medium 1x1 | 0.68      | 1.19                    | 1.15                         | 0.34           | 0.93   | PyTorch            |

Acceptance-band check:

- `conv_large`: PASS (`7.09` vs best external `8.70`, better than target).
- `conv_medium`: FAIL (best MMatrix `1.31` vs best external `0.64` → `2.05x`,
  target `<=1.25x`).
- `pw_medium`: FAIL (best MMatrix `0.68` vs best external `0.34` → `2.00x`,
  target `<=1.20x`).

Optimization priority from snapshot:

1. `conv_medium`: reduce submit/boundary overhead (focus on command-buffer
  lifecycle and host dispatch overhead, not kernel math).
2. `pw_medium` (1x1): keep/strengthen GEMM routing and avoid MPS dispatch for
  this workload class by default.
3. Preserve current `conv_large` behavior; avoid regressions while tuning
  medium/pointwise thresholds.

### P5: Implementation sprint for medium-gap closure

Objective:

- Reduce `conv_medium` host/submit/boundary gap without regressing
  `conv_large`.
- Keep 1x1 (`pw_medium`) on GEMM-default behavior unless MPS proves faster
  under the same benchmark contract.

Completed in this sprint (2026-05-16):

- Fused async output-write ordering fix in MPS backend:
  - `st_backend_conv2d_batchnorm2d_forward_mps`
  - `st_backend_conv2d_batchnorm2d_pool_forward_mps`
- Change: drain `output->buf->_async_cmd_buf` **before** encoding/committing a
  new write to the same output buffer (aligns fused path with conv fastpath
  rule, avoids overlapping writes/orphaned pending states).

Smoke validation after change (`bench_st_pipeline`, opt build):

- `conv+bn` medium: boundary-sync faster than sync-each in this run.
- `conv+bn` large: sync-each faster in this run.
- `conv+bn+pool` medium/large: sync-each faster in this run.

Interpretation:

- The ordering fix is retained for correctness/scheduling hygiene.
- Boundary policy for fused pipelines remains shape-sensitive; no global default
  flip yet.

Next implementation tasks:

1. Add variance-aware repeated pipeline measurement (fixed seed + median +
   p10/p90) before changing fused sync defaults.
2. Add explicit routing guard for pointwise 1x1 workloads in AUTO mode
   (documented by parity snapshot where GEMM beats MPS for `pw_medium`).
3. Re-run cross-framework parity after each routing/sync change and keep
   `conv_large` within current pass band.

## Do Not Do Yet

- Do not route `conv_medium` away from MPS solely because isolated single-op
  latency is worse than PyTorch/MLX.
- Do not assume `mps_true_async_boundary` is universally faster than
  `mps_zero_copy_sync`.
- Do not treat negative `cpu_overhead_ms` in batch mode as a real negative CPU
  cost.
- Do not introduce broad graph/session abstractions until the existing explicit
  sync policy is exhausted.
