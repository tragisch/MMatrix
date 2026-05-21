# MMatrix

MMatrix is a C project for matrix and tensor operations, built with Bazel.
It targets macOS and Linux, with CPU acceleration (BLAS/OpenMP) and optional MPS/Metal support on Apple Silicon.

## What is included

- `app/matrix`: dense (`sm`, `dm`) and sparse (`dms`) matrix modules
- `app/matrix:sm_mps`: explicit Apple Silicon MPS path for GPU-resident large
  dense GEMMs; check around `1024^3`, recommended for `2048^3`-class matrices.
- `app/tensor`: N-dimensional tensor module (`st`) with CPU and optional MPS backend (only on Apple Silicon).

## Build

Install common dependencies on macOS (Homebrew):

```bash
brew install openblas libomp suitesparse matio llvm
```

Build everything:

```bash
bazel build //...
```

Build specific modules:

```bash
bazel build //app/matrix
bazel build //app/tensor
```

Optional backend configs (matrix):

```bash
bazel build //app/matrix --config=accelerate
bazel build //app/matrix --config=openblas
bazel build //app/matrix --config=openmp_only
```

## Test

```bash
bazel test //...
```

Unit tests are in `app/matrix/tests/` and `app/tensor/tests/`.


## Documentation

Build docs:

`bazel build //:docs`

Serve docs locally:

`bazel run //:docs_serve`

Regenerate Tensor + Matrix API pages from public header docs (Doxygen XML -> Markdown):

`bazel run //:docs_refresh_api`

The short aliases `//:docs` and `//:docs_serve` are the recommended user entrypoints.

## Install artifacts

Install headers and libraries to a custom directory:

```bash
bazel run //:matrix_installer -- $(PWD)/lib/matrix
```

Modes:

```bash
bazel run //:matrix_installer -- --mode=standard $(PWD)/lib/matrix
bazel run //:matrix_installer -- --mode=bundle $(PWD)/lib/matrix_bundle
```

## Platforms

- macOS ARM64 (Apple Silicon)
- Linux x86-64

## License

MIT (see `LICENSE.txt`).
Third-party licenses are provided by the respective upstream projects.
