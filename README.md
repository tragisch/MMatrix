# MMatrix

A personal C library playground for matrix and tensor operations on macOS and Linux, with BLAS/Accelerate acceleration and an MPS/Metal backend for Apple Silicon.

---

## Modules

### `app/matrix` — Matrix library

| Module | Type | Description |
|--------|------|-------------|
| `sm` | `FloatMatrix` | Single-precision dense matrix |
| `dm` | `DoubleMatrix` | Double-precision dense matrix |
| `dms` | sparse COO | Double-precision sparse matrix (CSparse) |

All matrix modules support element-wise and matrix operations, PCG-based random creation, console printing, and I/O for MATLAB `.mat` and Matrix Market files.

### `app/tensor` — Tensor library

| Module | Type | Description |
|--------|------|-------------|
| `st` | `FloatTensor` | N-dimensional float/bf16 tensor |

Supported ops (CPU + MPS/Metal on macOS):

- Conv2D forward, fused Conv2D+BatchNorm2D forward
- MaxPool2D, AvgPool2D, BatchNorm2D forward
- GEMM-backed fallback for unsupported shapes

MPS dispatch is runtime-selectable; call `st_set_backend(ST_BACKEND_MPS)`.

---

## Performance Notes

- **macOS**: Apple Accelerate (vDSP, BLAS, LAPACK) + optional MPS
- **Linux**: OpenBLAS + OpenMP
- ARM NEON intrinsics on Apple Silicon

**MPS break-even (GEMM, measured):** MPS outperforms Accelerate only above roughly $3072\times3072$ matrices. Below that, GPU dispatch overhead dominates.

### Choosing a matrix format

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Dense / moderately sparse, N ≤ 10k | `sm.h` / `dm.h` | BLAS-accelerated, fits in memory |
| Sparse (density < 1%), N ≤ 20k | `dms.h` | COO + CSparse, low overhead |
| Very large sparse (N > 50k, nnz > 1M) | SuiteSparse:GraphBLAS | Parallel CSC kernels, semiring algebra |

---

## Build

Install dependencies via Homebrew:

```bash
brew install openblas libomp suitesparse matio llvm
```

Build everything:

```bash
bazel build //...
```

Or per module:

```bash
bazel build //app/matrix
bazel build //app/tensor
```

The default backend is selected automatically (Accelerate on macOS, OpenBLAS on Linux). Override when needed:

```bash
bazel build //app/matrix --config=accelerate    # macOS, explicit
bazel build //app/matrix --config=openblas      # force OpenBLAS
bazel build //app/matrix --config=openmp_only   # no BLAS fallback
```

---

## Tests

```bash
bazel test //...
```

Tests use [Unity](https://www.throwtheswitch.org/unity) and live in `app/matrix/tests/` and `app/tensor/tests/`.

---

## Quick Examples

```c
// Matrix
FloatMatrix *A = sm_create_random(128, 128);
FloatMatrix *B = sm_transpose(A);
sm_destroy(A);
sm_destroy(B);

// Tensor
size_t shape[] = {1, 3, 224, 224};
FloatTensor *x = st_create(4, shape);
st_destroy(x);
```

---

## Installation

Install the compiled library and headers into a custom directory:

```bash
bazel run //:matrix_installer -- $(PWD)/lib/matrix
```

Supports two modes:

```bash
bazel run //:matrix_installer -- --mode=standard $(PWD)/lib/matrix        # default
bazel run //:matrix_installer -- --mode=bundle   $(PWD)/lib/matrix_bundle  # single fat archive
```

---

## Supported Platforms

- macOS ARM64 (Apple Silicon M1/M2/M3+), Clang/LLVM
- Linux x86-64 (Ubuntu + Homebrew), Clang/LLVM with OpenMP

---

## Dependencies

All pulled via [Bazel](https://bazel.build) modules or [Homebrew](https://brew.sh):

- [OpenBLAS](https://www.openblas.net/)
- [libomp](https://openmp.llvm.org/)
- [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html)
- [matio](https://sourceforge.net/projects/matio/) + HDF5
- [PCG](https://www.pcg-random.org/)

---

## License

MIT License (see [LICENSE](LICENSE.txt)).
Includes files from the Bazel project under Apache License 2.0 (`tools/install/`).
