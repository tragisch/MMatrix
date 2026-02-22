---
name: sm-gemm-optimization
description: Optimize dense matrix multiplications in MMatrix using sm_gemm. Use when asked about GEMM performance, batching/stacking, transpose flags, bias+ReLU fusion, Accelerate/MPS behavior, or tuning matrix multiply hot paths in C.
license: MIT
---

# sm_gemm optimization

Use this skill to make `sm_gemm` fast and consistent across common workloads (small GEMMs, batched GEMMs, and large GEMMs).

## When to Use This Skill
- You are asked to speed up `sm_gemm` or GEMM-heavy code paths.
- You need guidance on batching/stacking many small GEMMs.
- You need to avoid costly transposes and extra allocations.
- You are using Accelerate or MPS and want break-even guidance.
- You want to fuse bias + ReLU into the GEMM path.

## Prerequisites
- Familiarity with `sm_gemm` and `SmTranspose` flags.
- Understanding of row-major layout in this project.

## Step-by-Step Workflows
1. **Check dimensions and flags**
   - Verify `C->rows`/`C->cols` match `A`, `B`, and transpose flags.
   - Prefer `SM_TRANSPOSE` flags instead of calling `sm_transpose`.

2. **Batch/stack small GEMMs**
   - If you have many small GEMMs, stack matrices (e.g., batch on rows).
   - Replace a loop of small GEMMs with a single larger GEMM when possible.

3. **Use fusion when applicable**
   - For `C = relu(A*B + bias)`, call `sm_gemm_bias_relu`.
   - Bias can be `1×cols` or `rows×cols`.

4. **Accumulate with beta**
   - If `C` already contains values, use the `beta` parameter to accumulate.

5. **MPS decision (USE_ACCELERATE_MPS)**
   - MPS is typically beneficial only for very large dimensions.
   - Break-even is roughly ~3072 per dimension in this repo’s measurements.

6. **Threading for Accelerate**
   - For small GEMMs, `VECLIB_MAXIMUM_THREADS=1` can be faster.
   - For large GEMMs, leave default threading.

## Troubleshooting
| Symptom | Likely Cause | Fix |
|---|---|---|
| GEMM is slow for small sizes | Too many tiny calls | Stack/batch into one GEMM |
| High memory churn | Explicit `sm_transpose` allocations | Use transpose flags |
| Extra pass for bias+ReLU | Separate ops | Use `sm_gemm_bias_relu` |
| MPS slower than expected | Sizes below break-even | Keep Accelerate/BLAS path |

## References
- `app/matrix/src/sm.c` (implementation)
- `app/matrix/include/sm.h` (public API)
