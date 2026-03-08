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

**MPS break-even (measured):** On Apple Silicon, MPS begins to outperform
Accelerate for GEMM only around roughly $3072\times3072$ square matrices
(real-time). Below that, Accelerate is typically faster due to GPU dispatch
overhead.

---

## Choosing the Right Matrix Format

| Scenario                                       | Recommended               | Reason                                                                                   |
| ---------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------- |
| Dense or moderately sparse matrix, N ≤ 10k     | **`sm.h`** / **`dm.h`**   | BLAS/Accelerate kernels are hard to beat; memory fits easily on a laptop                 |
| Sparse matrix (density < 1%), N up to ~20k     | **`dms.h`**               | COO + CSparse keeps overhead low; native SpGEMM with OpenMP scales well                  |
| Very large sparse matrices (N > 50k, nnz > 1M) | **SuiteSparse:GraphBLAS** | Parallel CSC kernels, semiring algebra, and optimized memory management pay off at scale |

**Rule of thumb:** For graphs with fewer than ~20k nodes, `dms.h` combined with
CSparse provides an excellent balance of simplicity and performance.  For dense
or high-density graphs under 10k nodes, prefer `sm.h` (float) or `dm.h`
(double) — the BLAS-accelerated dense path is faster and the memory footprint is
still manageable.  For truly large-scale sparse problems (100k+ nodes, millions
of non-zeros), consider [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS),
which is available as a Bazel dependency in this project (`@graphblas`).

---

## Documentation

- Overview and API style: [`docs/Overview.md`](docs/Overview.md)
- Performance notes: [`docs/Performance.md`](docs/Performance.md)
- Grafana benchmark workflow: [`docs/Grafana.md`](docs/Grafana.md)
- Float dense API (`sm`): [`docs/api/sm.md`](docs/api/sm.md)
- Double dense API (`dm`): [`docs/api/dm.md`](docs/api/dm.md)
- Sparse COO API (`dms`): [`docs/api/dms.md`](docs/api/dms.md)

  
---

## Build

Ensure required dependencies are installed via Homebrew (Linux too):

```bash
brew install openblas libomp suitesparse matio llvm
```

Then build using Bazel on macOS:

```bash
bazel build //app/matrix
```
On Linux use '--config=linux_llvm' to use the LLVM toolchain installed by Homebrew:

```bash
bazel build //app/matrix --config=linux_llvm
```

### Options

You can enable optional BLAS backends via Bazel defines:

- **Linux** and **macOS**:
  ```bash
  bazel build //app/matrix --define=USE_OPENBLAS=1
  ```

- **macOS**:
  - Use Accelerate framework (includes MPS support):
    ```bash
    bazel build //app/matrix --define=USE_ACCELERATE=1
    ```
    MPS (Metal Performance Shaders) is built in automatically on macOS.
    To activate MPS at runtime, call `sm_set_backend(SM_BACKEND_MPS)` in your code.

---
## Installation

You can install the compiled library and headers into a custom directory using the Bazel installer target:

```bash
bazel run //:matrix_installer -- /your/installation/path
```
i.e. 

```bash
bazel run //:matrix_installer -- $(PWD)/lib/matrix
```
The installer supports two modes:

- `standard` (default): installs `libmatrix.a` plus dependency libs and headers.
- `bundle`: creates and installs one combined static archive `libmatrix_full.a` (fat static lib) and installs headers.

Examples:

```bash
# Default mode
bazel run //:matrix_installer -- --mode=standard $(PWD)/lib/matrix

# Bundle mode (single static archive)
bazel run //:matrix_installer -- --mode=bundle $(PWD)/lib/matrix_bundle
```

To also create symbolic links into system-wide directories (e.g., `/usr/local/include`, `/usr/local/lib`), you can enable system integration in BUILD - File:

```bash
installer(
    name = "matrix_installer",
    data = [
        "//app/matrix",        # the target to be installed
        "//app/matrix:matrix_header", # collected in a Bazel filegroup with all public headers
        "//:LICENSE.txt",      # if available
        "@libomp//:install_files",
        "@log",
        "@matio",
        "@openblas//:install_files",
        "@pcg",
        "@suitesparse//:install_files",
        "@zlib",
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
bazel test //...
```

Tests are located in the `app/matrix/tests/` folder. You can add new test files and they will be discovered automatically by Bazel.

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
