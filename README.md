# MMatrix

**MMatrix** is my personal C matrix library playground for macOS and Linux, supporting both dense and sparse matrix formats with SIMD and BLAS acceleration.

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

### Performance

- [macOS] Optimized with Apple Accelerate (vDSP, BLAS, LAPACK)
- [macOS] Optional use of Metal Performance Shaders (MPS)
- SIMD acceleration with OpenMP and [macOS] ARM NEON intrinsics
- PCG-based random number generation for reproducibility and parallelism
  
---

## Build

Ensure required dependencies are installed via Homebrew (Linux too):

```bash
brew install openblas libomp suitesparse matio llvm
```

Then build using Bazel on macOS:

```bash
bazel build //src:matrix
```
On Linux use '--config=linux_llvm' to use the LLVM toolchain installed by Homebrew:

```bash
bazel build //src:matrix --config=linux_llvm
```

### Options

You can enable optional BLAS backends via Bazel defines:

- **Linux** and **macOS**:
  ```bash
  bazel build //src:matrix --config=linux_llvm --define=USE_OPENBLAS=1
  ```

- **macOS**:
  - Use Accelerate framework:
    ```bash
    bazel build //src:matrix --define=USE_ACCELERATE=1
    ```
  - Use Accelerate + Metal (MPS):
    ```bash
    bazel build //src:matrix --define=USE_ACCELERATE_MPS=1
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
        "//src:matrix_header", # collected in a Bazel filegroup with all public headers
        "//:LICENSE.txt",      # if available
        ],
    system_integration = False, # set "True" symlinks into /usr/local/lib and /usr/local/include
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

Tests are located in the `tests/` folder. You can add new test files and they will be discovered automatically by Bazel.

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
- Linux (x86 64bit, tested on Ubuntu + Homebrew)
- Clang/LLVM toolchain with OpenMP support required

---

## Dependencies

This project depends on:

- [OpenBLAS](https://www.openblas.net/)
- [libomp](https://openmp.llvm.org/)
- [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html)
- [matio](https://sourceforge.net/projects/matio/)
- [LLVM](https://llvm.org/) (via Homebrew)

All dependencies are pulled via [Bazel](https://bazel.build) modules or expected to be installed via [Homebrew](https://brew.sh).

---


## License
This project is primarily licensed under the MIT License (see LICENSE).
It includes files from the Bazel project, licensed under the Apache License 2.0 (see tools/install/*, ).