# MMatrix

**MMatrix** is my personal C matrix library playground for macOS, supporting both dense and sparse matrix formats with SIMD and BLAS acceleration.

---

## Features

- `sm`: Single-precision (float) dense matrix
- `dm`: Double-precision (double) dense matrix
- `dms`: Double-precision sparse matrix (COO format)

All modules support:

- Random matrix creation using PCG (permuted congruential generator)
- Elementwise and matrix operations
- Console-based printing
- I/O for:
  - MATLAB `.mat` files (via `matio` and `hdf5`)
  - Matrix Market files

---

## Performance

Currently:
- Optimized with Apple Accelerate (vDSP, BLAS, LAPACK)
- Optional use of Metal Performance Shaders (MPS)
- SIMD acceleration with OpenMP and ARM NEON intrinsics
- PCG-based random number generation for reproducibility and parallelism

---

## Build

Ensure required dependencies are installed via Homebrew:

```bash
brew install openblas libomp suitesparse matio
```

Then build using Bazel:

```bash
bazel build //src:matrix
```

---

## Run Tests

Unit tests are written using [Unity](https://www.throwtheswitch.org/unity):

```bash
bazel test //tests:all
```

---

## Quick Example

```c
FloatMatrix *A = sm_create_random(128, 128);
FloatMatrix *B = sm_transpose(A);
sm_destroy(A);
sm_destroy(B);
```

---

## Supported Platforms

- macOS (ARM64 preferred, supports Apple Silicon M1/M2/M3 ...)
- Clang toolchain with OpenMP support required

---

## License

MIT – see `LICENSE` file.
