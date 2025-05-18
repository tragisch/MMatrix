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
