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