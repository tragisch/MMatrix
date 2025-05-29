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
  
My Mac:
MacBook Air M3 (24GB)  

| Library sm_multiply()                    | 50x50     | 64x64     | 128x128   | 256x256   | 512x512   | 1024x1024   | 2048x2048   | 4096x4096   | 5000x5000   | RMS       | BigO  |
|:--------------------------|:----------|:----------|:----------|:----------|:----------|:------------|:------------|:------------|:------------|:----------|:------------------|
| Apple Accelerate          | 1.59 µs   | 1.28 µs   | 4.25 µs   | 24.43 µs  | 166.80 µs | 1.51 ms     | 12.25 ms    | 99.57 ms    | 190.57 ms   | 4.55e-11  | 0.00151 × N^3     |
| Metal Performance Shaders | 1.68 µs   | 1.31 µs   | 4.17 µs   | 24.40 µs  | 166.60 µs | 1.51 ms     | 5.49 ms     | 23.23 ms    | 31.92 ms    | 7.33e-11  | 1.31041 × N^2     |
| Naive                     | 177.63 µs | 366.40 µs | 2.89 ms   | 22.58 ms  | 179.10 ms |             |             |             |             | 4.15e-13  | 1.33566 × N^3     |
| No BLAS, ARM NEON         | 14.66 µs  | 16.91 µs  | 37.10 µs  | 171.77 µs | 1.16 ms   | 9.77 ms     | 111.41 ms   | 1.48 s      | 3.20 s      | 1.66e-10  | 0.02464 × N^3     |
| OpenBLAS                  | 3.52 µs   | 14.89 µs  | 119.13 µs | 211.83 µs | 1.00 ms   | 7.37 ms     | 42.95 ms    | 289.20 ms   | 478.63 ms   | 9.10e-11  | 0.00392 × N^3     |

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
## Installation

You can install the compiled library and headers into a custom directory using the Bazel installer target:

```bash
bazel run //:matrix_installer -- /your/installation/path
```

To also create symbolic links into system-wide directories (e.g., `/usr/local/include`, `/usr/local/lib`), you can enable system integration in BUILD - File:

```bash
installer(
    name = "matrix_installer",
    data = [
        "//src:matrix",        # the target to be installed
        "//src:matrix_header", # must be collected in a filegroup
        "//:LICENSE.txt",      # if available
        ],
    system_integration = False, # set "True" symlink to /usr/local/lib and /usr/local/include
)
```

This will:

- Copy the compiled static library (`libmatrix.a`) and dependencies into `lib/`
- Install public headers into `include/`
- Optionally create symlinks for easy access from standard system locations


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
This project is primarily licensed under the MIT License (see LICENSE).
It includes files from the Bazel project, licensed under the Apache License 2.0 (see tools/bazel/install/*, ).
