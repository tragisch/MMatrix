# Changelog

## 0.4.0 (unreleased)

Changes since `v0.3.1`.

### Added

- Vector modules for float/double, strided views, sparse vectors, geometry, and vector/tensor bridges.
- Flat-buffer SGD, momentum, RMSProp, Adam, and gradient norm clipping kernels.
- Shared NN activation/backward kernels, fused GEMM + bias + activation, and row/loss kernels.
- Tensor storage abstraction, Metal buffer support, and f32/bf16 conversion utilities.
- Interpreter kernels: broadcast binary arithmetic, LayerNorm, ReduceMax, batched matmul, gather, and materializing permute.
- IEEE binary16 scalar/bulk conversion helpers.
- GPU-resident matrix GEMM streams/plans and reproducible benchmark sources.

### Changed / migration

- SuiteSparse is no longer a library or build dependency. Sparse matrices continue to use COO with a lazy CSR row cache.
- Removed `dms_to_cs`, `dms_from_cs`, and `cs_to_dms`, and the CSparse cache fields in `DoubleSparseMatrix`. Code accessing those APIs/fields must migrate; recompile consumers.
- Matrix/tensor conversions now live in `st_convert.h` and `//app/tensor:st`, rather than `m_convert.h`. `st_from_sm` creates a rank-2 `[rows, cols]` tensor; `sm_from_st` requires a rank-2 f32 tensor.
- Tensor implementation files are grouped into core, backend, runtime, and operation directories. Depend on public Bazel targets rather than source paths.
- Bazel version and module lockfile are included for reproducible dependency resolution. Linux OpenBLAS remains a system installation.

### Fixed during release preparation

- Exercise actual OpenMP branches in interpreter thread-comparison tests and remove an unmatched floating-point pragma.
- Reject size overflows before gather/permute byte copies; validate permutation indices before shape access; preserve arbitrary unaligned byte payloads.
- Remove stale SuiteSparse and local HDF5 wiring, obsolete vendor build wrappers, and benchmark targets whose sources were not published.
- Include the MPS benchmark helper header in the source release and exclude local experiments from Bazel discovery.
- Install transitive library archives and required external headers; include tensor artifacts on macOS; use portable macOS/Linux install flags.
- Add macOS/Ubuntu CI coverage with standard/bundle installation consumer checks.
- Repair CI test discovery and document the platform boundaries and new API entrypoints.
