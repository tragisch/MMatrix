# rmatrix

Safe Rust wrapper for the `FloatMatrix` API provided by the C-based mMatrix library.

## Building with Bazel

The Bazel target `//app/rmatrix:rmatrix` produces the Rust library that links against
`//app/matrix:sm`. A companion `rust_test` target (`//app/rmatrix:rmatrix_tests`) runs
the embedded smoke tests.

```bash
bazel build //app/rmatrix:rmatrix
bazel test //app/rmatrix:rmatrix_tests
```

### Native Dependencies

The wrapper relies on the existing C targets. Ensure the desired feature flags (for
Accelerate, OpenBLAS, etc.) are provided the same way you build the C library, e.g.

```bash
bazel test //app/rmatrix:rmatrix_tests --define=USE_ACCELERATE_MPS=1
```

### Toolchain Configuration

Bzlmod now pulls in `rules_rust` and registers a stable toolchain (1.76.0). If you
need a different Rust version or features, adjust the `rust.toolchain` stanza in
`MODULE.bazel`.
