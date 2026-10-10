# nm_lib Kernel Migration Map

This checklist maps scalar kernels in `14_NN/app/nm_lib` to the reusable
mMatrix APIs. Phase 1 covers dense NN kernels; interpreter-only tensor
operators remain a separate phase.

| nm_lib function / operation | mMatrix function |
| --- | --- |
| `nm_relu` | `sm_apply_relu`, existing `st_apply_relu` |
| `nm_sigmoid` | `sm_apply_sigmoid`, `st_apply_sigmoid` |
| `nm_tanh` | `sm_apply_tanh`, `st_apply_tanh` |
| `nm_softmax` | `sm_apply_softmax`, `st_apply_softmax` |
| Interpreter tanh GELU | `sm_apply_gelu_tanh`, `st_apply_gelu_tanh` |
| Interpreter erf GELU | `sm_apply_gelu_erf`, `st_apply_gelu_erf` |
| `nm_sigmoid_grad` | `sm_apply_sigmoid_backward`, `st_apply_sigmoid_backward` |
| `nm_tanh_grad` | `sm_apply_tanh_backward`, `st_apply_tanh_backward` |
| `nm_relu_grad` | `sm_apply_relu_backward`, existing `st_apply_relu_backward` |
| Interpreter GELU gradient | `sm_apply_gelu_*_backward`, `st_apply_gelu_*_backward` |

## Accuracy and determinism

Accelerate uses vector vForce math where available. Its results can differ
slightly from scalar libm, so activation comparisons use relative tolerance
$10^{-5}$ plus absolute tolerance $10^{-6}$. The erf GELU uses the standard
erf definition; the tanh GELU uses the common cubic approximation.

Elementwise work is partitioned by index and is independent of OpenMP thread
count. Softmax and row reductions keep each row/column sum in a fixed order to
avoid thread-count-dependent accumulation. BF16 tensor activation kernels
promote to F32, compute, then round back to BF16.

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