# MMatrix

MMatrix is a C library for dense and sparse matrix, vector, and tensor operations, built with Bazel.
The next release candidate is **0.4.0**; see [CHANGELOG.md](CHANGELOG.md) for changes and migration notes.

## What is included

- `app/matrix`: dense float (`sm`), dense double (`dm`), sparse COO (`dms`), matrix conversions, and MAT/MatrixMarket I/O.
- `app/vector`: dense (`sv`, `dv`), strided (`vv`), sparse (`dvs`), fixed-size geometry (`vec3`), and flat-buffer optimizer kernels (`sv_optim`).
- `app/tensor`: N-dimensional f32/bf16 tensors, shape/view operations, convolution, pooling, BatchNorm, and graph-interpreter kernels (broadcast arithmetic, LayerNorm, ReduceMax, batched matmul, gather, and materializing permute). Binary16 conversion is a separate raw-buffer API, not a tensor storage dtype.
- `app/matrix:sm_mps`: explicit Apple Silicon MPS path for GPU-resident dense GEMMs, with streams and reusable plans.

## Platforms and dependencies

| Platform | Modules | Default CPU backend |
| --- | --- | --- |
| macOS ARM64 (Apple Silicon) | Matrix, vector, tensor; optional MPS/Metal execution | Accelerate + OpenMP |
| Linux x86-64 | Matrix and vector | OpenBLAS + OpenMP |

The tensor target currently requires macOS frameworks, including for CPU execution.
Linux support does not include the tensor module. Release validation status is recorded in [doc/release-0.4.0.md](doc/release-0.4.0.md).

Install Bazelisk and the platform's C/C++ development tools. The repository pins Bazel in `.bazelversion`.
On macOS, install Xcode command line tools (`xcode-select --install`). No Homebrew SuiteSparse installation is required.

Bzlmod resolves LLVM, OpenMP, matio, zlib, and the test/documentation tooling through the configured registries.
`log.c` is included under `third_party/log.c`. OpenBLAS is a **system dependency** when selecting the OpenBLAS backend:

```bash
# Homebrew, including Linuxbrew:
brew install openblas
# For another installation prefix, pass it explicitly:
bazel build //app/matrix --config=openblas --repo_env=OPENBLAS_ROOT=/path/to/openblas
```

The default OpenBLAS prefixes are `/opt/homebrew/opt/openblas` on macOS and `/home/linuxbrew/.linuxbrew/opt/openblas` on Linux.
The OpenBLAS prefix must expose headers under `include/` and `libopenblas.a` under `lib/`.
LAPACKE must be included in OpenBLAS or supplied as `lib/liblapacke.a`; the Ubuntu CI job stages the distribution packages into this layout.
SuiteSparse and the former CSparse conversion API have been removed.

## Build and test

```bash
bazel build //...
bazel test //...
```

Build individual modules:

```bash
bazel build //app/matrix //app/vector
# macOS only:
bazel build //app/tensor
```

Matrix backend configs:

```bash
bazel build //app/matrix --config=accelerate  # macOS
bazel build //app/matrix --config=openblas   # Linux
bazel build //app/matrix --config=openmp_only
```

Unit tests live alongside each module in `app/matrix/tests/`, `app/vector/tests/`, and `app/tensor/tests/`.
Benchmarks under `app/*/tests/` and the published MPS benchmark under `share/simple_benchmark/` are buildable targets.
Local vendor experiments and legacy benchmarks are excluded from recursive discovery with `.bazelignore`.

## Documentation

```bash
bazel build //:docs
bazel run //:docs_serve
```

Regenerate matrix and tensor API pages from public header comments (requires Doxygen and Python 3 on PATH):

```bash
bazel run //:docs_refresh_api
```

## Install artifacts

Install public headers and libraries, including their link dependencies, to a custom directory:

```bash
bazel run //:matrix_installer -- "$PWD/lib/matrix"
bazel run //:matrix_installer -- --mode=bundle "$PWD/lib/matrix_bundle"
```

Standard mode keeps separate libraries; bundle mode combines static archives into `libmatrix_full.a`.
On macOS the installation includes tensor and MPS libraries; Linux installs matrix and vector libraries.
System frameworks still need to be linked by consumers. On Linux, link separate static archives as a dependency group (`-Wl,--start-group ... -Wl,--end-group`) together with the system C++, Fortran, math, threading, and dynamic-loader runtimes; `tools/ci/test_install.sh` contains a complete example. Installation does not create a portable binary SDK:
compiler/runtime compatibility and third-party notices must be checked before redistributing binary bundles.

## License

MIT (see `LICENSE.txt`). Third-party dependencies retain their upstream licenses.
