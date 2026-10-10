# nm_lib Kernel Migration Map

This checklist maps scalar kernels in `14_NN/app/nm_lib` to the reusable
mMatrix APIs. Phase 1 covers dense NN kernels; Phase 2 covers the
graph-interpreter tensor operators.

| nm_lib function / operation | mMatrix function |
| --- | --- |
| `nm_relu` | `sm_apply_relu`, existing `st_apply_relu` |
| Dense GEMM + bias + activation | `sm_gemm_bias_act` |
| `nm_sigmoid` | `sm_apply_sigmoid`, `st_apply_sigmoid` |
| `nm_tanh` | `sm_apply_tanh`, `st_apply_tanh` |
| `nm_softmax` | `sm_apply_softmax`, `st_apply_softmax` |
| Interpreter tanh GELU | `sm_apply_gelu_tanh`, `st_apply_gelu_tanh` |
| Interpreter erf GELU | `sm_apply_gelu_erf`, `st_apply_gelu_erf` |
| `nm_sigmoid_grad` | `sm_apply_sigmoid_backward`, `st_apply_sigmoid_backward` |
| `nm_tanh_grad` | `sm_apply_tanh_backward`, `st_apply_tanh_backward` |
| `nm_relu_grad` | `sm_apply_relu_backward`, existing `st_apply_relu_backward` |
| Interpreter GELU gradient | `sm_apply_gelu_*_backward`, `st_apply_gelu_*_backward` |
| `nm_softmax_ce_grad` | `sm_softmax_ce_grad` |
| `nm_loss_mse` | `sm_loss_mse` |
| `nm_loss_cross_entropy` | `sm_loss_cross_entropy` |
| `nm_add_rowwise` | `sm_add_rowwise` |
| `nm_sum_rows` | `sm_sum_rows` |
| `nm_argmax_rows` | `sm_argmax_rows` |
| `optimizer_apply_array` (SGD) | `sv_optim_sgd` |
| `optimizer_apply_array` (Momentum) | `sv_optim_momentum` |
| `optimizer_apply_array` (RMSprop) | `sv_optim_rmsprop` |
| `optimizer_apply_array` (Adam) | `sv_optim_adam` |
| Gradient L2 clipping | `sv_clip_norm` |
| Interpreter `run_binary` (add/sub/mul/div) | `st_binary_broadcast`, `st_binary_broadcast_into` |
| Interpreter `run_layernorm` | `st_layernorm` |
| Interpreter `run_reducemax` | `st_reduce_max`, `st_reduce_max_into` |
| Interpreter `run_matmul` | `st_matmul_batched` |

## Accuracy and determinism

Accelerate uses vector vForce math where available. Its results can differ
slightly from scalar libm, so activation comparisons use relative tolerance
$10^{-5}$ plus absolute tolerance $10^{-6}$. The erf GELU uses the standard
erf definition; the tanh GELU uses the common cubic approximation.

Elementwise work is partitioned by index and is independent of OpenMP thread
count. Softmax and row reductions keep each row/column sum in a fixed order to
avoid thread-count-dependent accumulation. BF16 tensor activation kernels
promote to F32, compute, then round back to BF16.

The interpreter broadcast kernels (`st_binary_broadcast*`) are bit-exact
against the scalar reference for all four operations, including IEEE results
for division by zero. Two caveats under the vector packages' `-ffast-math`:

* `vDSP_vadd`/`vsub`/`vmul` are single IEEE operations and used on the
  Accelerate path; `vDSP_vdiv` differs from scalar division by 1 ulp, so
  division always runs on the C path.
* `-ffast-math` rewrites a division by a loop-invariant scalar into a
  multiply with a once-rounded reciprocal (double rounding). The scalar
  operand path uses an IEEE-exact division helper compiled with
  `#pragma float_control(precise, on)` (clang) or
  `__attribute__((optimize("no-fast-math")))` (gcc).

## Broadcast binary benchmark

`//app/tensor:bench_st_binary` replicates nm_lib's `run_binary` (walk plus
row kernels) as the `before` baseline and compares medians for transformer
shapes (default OpenMP thread count was 4 on this 4-performance-core Apple
Silicon machine; with the thread threshold below, default and single-thread
runs take the same sequential path):

| Case | Threads | Elements | nm_lib before | `st_binary_broadcast_into` | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| same-shape add [8,256,512] | 1 | 1,048,576 | 389.500 | 77.025 | 5.06× |
| bias add [4,128,256]+[256] | 1 | 131,072 | 60.150 | 31.550 | 1.91× |
| bias add [4,512,768]+[768] | 1 | 1,572,864 | 1,077.750 | 663.125 | 1.63× |
| attn mask add (walk) [2,4,128,128]+[1,1,128,128] | 1 | 131,072 | 59.975 | 39.200 | 1.53× |
| attn mask add (suffix) [2,4,128,128]+[128,128] | 1 | 131,072 | 70.950 | 37.500 | 1.89× |
| scale mul [4,128,256]*[1] | 1 | 131,072 | 46.375 | 18.925 | 2.45× |
| scale mul (mid) [4,1,256]*[1,128,1] | 1 | 131,072 | 48.975 | 16.800 | 2.92× |
| suffix div [4,128,256]/[256] | 1 | 131,072 | 49.625 | 20.550 | 2.41× |
| attn mask add (walk) [1,4,16,16]+[1,1,16,16] | 1 | 1,024 | 0.628 | 0.760 | 0.83× |

The 1,024-element walk case is the fixed API overhead (validation and shape
checks) on a sub-microsecond kernel; larger walk inputs are 1.5× or better.

## LayerNorm and ReduceMax

`//app/tensor:bench_st_norm_reduce` replicates nm_lib's `run_layernorm` and
`run_reducemax` as the `before` baseline (medians on the same machine;
default OpenMP thread count 4). `st_layernorm` is the same algorithm as the
reference, so single-threaded it matches nm_lib; above the thread threshold
it parallelizes over groups. `st_reduce_max` folds trailing reductions with
vectorized run kernels; non-trailing masks walk like the reference plus one
vectorized NaN pre-scan.

| Case | Threads | Elements | nm_lib before | `st` kernel | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| layernorm [4,128,256] | 1 | 131,072 | 83.100 | 81.100 | 1.02× |
| layernorm [8,256,512] | default | 1,048,576 | 634.700 | 224.275 | 2.83× |
| layernorm [4,512,768] | default | 1,572,864 | 970.250 | 378.500 | 2.56× |
| layernorm [16,65,32] rank 2 | default | 33,280 | 20.547 | 21.418 | 0.96× |
| reduce max [2,4,128,128] axis 3 | 1 | 131,072 | 97.775 | 39.300 | 2.49× |
| reduce max [1,4,128,128] axis 3 | 1 | 65,536 | 50.562 | 19.738 | 2.56× |
| reduce max [4,128,256] axis 2 | 1 | 131,072 | 109.825 | 36.850 | 2.98× |
| reduce max [4,128,256] axis 1 (walk) | 1 | 131,072 | 18.250 | 44.725 | 0.41× |
| reduce max [2,1000] axis 0 (walk) | default | 2,000 | 0.270 | 0.719 | 0.38× |

### Norm and reduce thresholds

`st_layernorm` parallelizes over groups only above 262,144 input elements
(`ST_LAYERNORM_OMP_THRESHOLD`): at and below 262k the sequential kernel
matches the reference, and at 1M+ the parallel gain is 2.5× or better.
`st_reduce_max` parallelizes over output slots only above 1,048,576 input
elements (`ST_REDUCE_OMP_THRESHOLD`): measurements at 65k and 131k showed
no gain from a team on this machine, so realistic sizes stay sequential
(same rationale as the Phase 1 optimizer threshold). Both should be
retuned on the deployment machine.

### LayerNorm accuracy and NaN policy in ReduceMax

`st_layernorm` reproduces nm_lib's sum8 semantics exactly: the mean and the
variance use eight deterministic partial sums, so results are bit-identical
across thread counts; against a naive scalar loop the tolerance is 1e-5
relative plus 1e-6 absolute (and for groups of eight or fewer elements the
sum8 order is bit-identical to the naive order).

### Batched matmul

`st_matmul_batched` routes every product through the same backend as
`sm_gemm`, so compared with nm_lib's `run_matmul` loop (which also calls
`sm_gemm` per batch) the wall time is equivalent by construction;
`//app/tensor:bench_st_matmul` confirms this (medians, µs/call, default
thread count 4):

| Case | Threads | nm_lib before | `st_matmul_batched` | Speedup |
| --- | ---: | ---: | ---: | ---: |
| proj stacked [4,128,256]x[256,512] | 1 | 192.200 | 199.825 | 0.96× |
| proj stacked [8,256,512]x[512,512] | 1 | 1,923.750 | 1,914.575 | 1.00× |
| attn scores [2,4,128,64]x[2,4,64,128] | default | 45.000 | 45.175 | 1.00× |
| batched [4,12,64,64]x[4,12,64,64] | default | 81.925 | 80.400 | 1.02× |

The kernel's value is the unified API: NumPy batch broadcast, the
single-matrix stacking fast path, bf16 promote, and shape validation with
no partial writes. Batches are distributed over OpenMP only when there
are at least 8 of them and each GEMM is at most 262,144 elements
(`ST_MATMUL_OMP_MIN_BATCHES`/`ST_MATMUL_OMP_MAX_GEMM`), so BLAS-internal
threads are never oversubscribed. On this machine Accelerate serializes
small GEMMs, so the team measured neutral there; the path exists for
backends that benefit (for example OpenBLAS on Linux) and should be
re-measured on the deployment machine.

`st_reduce_max` deviates deliberately from the nm_lib reference for NaNs:
the reference's `x > max` fold silently drops NaNs, while `st_reduce_max`
propagates the bit pattern of the first NaN in scan order to the output
slot, so errors are not swallowed (documented in the header and covered by
tests). Finite values, ±INFINITY, and the -INFINITY start value follow the
reference. Under `-ffast-math` the fold therefore cannot use the OpenMP
`max` reduction (its private lanes start at -FLT_MAX and would swallow
-INFINITY) nor a horizontal vector max with compiler-inserted padding
lanes; the eight-lane fold combines through a noinline helper so all eight
lanes come from memory.

MatMul is the one interpreter kernel that is not bit-exact: the BLAS
accumulation order defines the result, so tests compare against a naive
reference (and across thread counts) with 1e-4 relative tolerance.

### Binary OpenMP threshold

The benchmark's thread sweep measures `st_binary_broadcast_into` at one
versus default threads (median µs/call):

| Elements | Path | 1 thread | default (4) | parallel gain |
| ---: | --- | ---: | ---: | ---: |
| 4,096 | same-shape | 1.409 | 2.350 | 0.60× |
| 16,384 | same-shape | 7.107 | 11.805 | 0.60× |
| 65,536 | same-shape | 5.018 | 4.688 | 1.07× |
| 262,144 | same-shape | 24.525 | 24.625 | 1.00× |
| 1,048,576 | same-shape | 124.300 | 236.050 | 0.53× |
| 2,097,152 | same-shape | 2,280.125 | 2,064.250 | 1.10× |
| 4,194,304 | same-shape | 1,422.750 | 1,280.500 | 1.11× |
| 8,388,608 | same-shape | 2,583.000 | 2,175.875 | 1.19× |
| 4,194,304 | scalar | 1,633.500 | 779.125 | 2.10× |
| 8,388,608 | scalar | 1,850.000 | 1,207.875 | 1.53× |

Parallel regions regress or stay flat below roughly one million elements
(OpenMP fork/join cost dominates a memory-bound kernel) and only gain
modestly above two million. `ST_BIN_OMP_THRESHOLD` is therefore 2,097,152
output elements, which keeps every tested transformer shape on the
sequential path; this follows the same rationale as the optimizer
threshold in Phase 1 and should be retuned on the deployment machine.

## Activation benchmark

The benchmark compares scalar reference loops (`before`) with the mMatrix
matrix/tensor wrappers (`after`). It prints medians for 256 through 1M elements,
with one thread and the default OpenMP thread count. The table below shows the
`tanh` case from an Apple Silicon Accelerate build (times are median µs/call;
default OpenMP thread count was 4).

| Elements | Threads | Scalar before | `sm_apply_tanh` | Speedup | `st_apply_tanh` | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 2.680 | 0.559 | 4.80× | 0.572 | 4.69× |
| 1,024 | 1 | 9.776 | 1.603 | 6.10× | 1.708 | 5.73× |
| 10,000 | 1 | 114.867 | 19.629 | 5.85× | 22.133 | 5.19× |
| 100,000 | 1 | 1,107.156 | 187.500 | 5.90× | 219.094 | 5.05× |
| 1,000,000 | 1 | 10,862.750 | 1,801.000 | 6.03× | 1,578.750 | 6.88× |
| 256 | 4 | 2.263 | 0.402 | 5.63× | 0.386 | 5.86× |
| 1,024 | 4 | 10.060 | 2.159 | 4.66× | 2.252 | 4.47× |
| 10,000 | 4 | 108.777 | 17.047 | 6.38× | 15.004 | 7.25× |
| 100,000 | 4 | 1,095.938 | 158.938 | 6.90× | 202.344 | 5.42× |
| 1,000,000 | 4 | 11,149.250 | 1,622.000 | 6.87× | 1,501.250 | 7.43× |

The benchmark executable also reports sigmoid, both GELU variants, and softmax.
Small sub-microsecond results are averaged over repeated calls within each
sample; the displayed values are medians of nine samples.

## Fused GEMM benchmark

The fused kernel was compared against GEMM, a row-bias loop, and activation
separately. Entries are median µs/call for ReLU on the same Apple Silicon
Accelerate build.

| Shape | Threads | Split before | `sm_gemm_bias_act` | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 32×32 | 1 | 5.910 | 1.220 | 4.84× |
| 64×64 | 1 | 25.500 | 3.708 | 6.88× |
| 128×128 | 1 | 111.500 | 16.167 | 6.90× |
| 256×256 | 1 | 505.000 | 153.000 | 3.30× |
| 32×32 | 4 | 6.950 | 1.220 | 5.70× |
| 64×64 | 4 | 21.917 | 3.750 | 5.84× |
| 128×128 | 4 | 107.167 | 16.500 | 6.49× |
| 256×256 | 4 | 493.000 | 87.000 | 5.67× |

## Row-helper benchmark

Median µs/call on the same machine, comparing scalar row loops with mMatrix
helpers (1024×256 and 4096×256 cases):

| Operation | Shape | Threads | Scalar before | mMatrix after | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `add_rowwise` | 1024×256 | 1 | 1,422.250 | 56.750 | 25.06× |
| `add_rowwise` | 1024×256 | 4 | 1,374.000 | 60.500 | 22.71× |
| `sum_rows` | 1024×256 | 1 | 1,735.250 | 491.000 | 3.53× |
| `sum_rows` | 1024×256 | 4 | 1,516.250 | 464.250 | 3.27× |
| `argmax_rows` | 1024×256 | 1 | 1,596.250 | 352.250 | 4.53× |
| `argmax_rows` | 1024×256 | 4 | 1,592.750 | 256.250 | 6.22× |
| `add_rowwise` | 4096×256 | 1 | 5,726.750 | 253.500 | 22.59× |
| `add_rowwise` | 4096×256 | 4 | 5,232.750 | 258.000 | 20.28× |
| `sum_rows` | 4096×256 | 1 | 8,220.500 | 3,115.750 | 2.64× |
| `sum_rows` | 4096×256 | 4 | 8,619.500 | 1,383.250 | 6.23× |
| `argmax_rows` | 4096×256 | 1 | 6,444.250 | 1,134.750 | 5.68× |
| `argmax_rows` | 4096×256 | 4 | 6,001.250 | 735.250 | 8.16× |

## Optimizer parallelization

The optimizer elementwise loops use OpenMP only for `n > 1,048,576`. Adam bias
correction and all update formulas match `optimizer_apply_array`; optional
decoupled weight decay is disabled when its argument is zero. `sv_clip_norm`
computes its norm in a fixed scalar order and parallelizes only the independent
scaling pass.

`//app/vector:bench_sv_optim` compares scalar-reference and API medians for
SGD, momentum, RMSprop, Adam, and clipping at 1k, 10k, 100k, and 1M elements,
with one and default OpenMP thread counts. A 16,384 threshold regressed several
optimizers at 100k. A follow-up trial at 262,144 activated a team at 1M but also
regressed most optimizers. The selected threshold is therefore above the
largest measured input, keeping tested models out of parallel regions. The
behavior beyond 1M remains unmeasured and should be retuned on the deployment
machine if larger parameter buffers are common.

Selected medians from the 262,144-threshold trial at 1M elements
(microseconds/call; default thread count 4). `API` uses the thresholded
optimizer; `forced OMP` is the parallel reference loop:

| Optimizer | Scalar | API (262k) | Forced OMP | Forced speedup |
| --- | ---: | ---: | ---: | ---: |
| SGD | 121.000 | 124.000 | 110.000 | 1.10× |
| Momentum | 201.000 | 429.000 | 153.000 | 1.31× |
| RMSprop | 220.000 | 266.000 | 156.000 | 1.41× |
| Adam | 467.000 | 408.000 | 316.000 | 1.48× |
| Clip norm | 137.000 | 127.000 | 115.000 | 1.19× |

The API-level OpenMP path still regressed momentum and RMSprop in this trial,
so the final threshold was raised above 1M. The direct parallel reference shows
potential benefit for later tuning, but does not justify parallelizing the
tested small-network range today.