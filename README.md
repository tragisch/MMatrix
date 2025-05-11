# MMatrix

A matrix C-library for MacOS. It is a wrapper for my purpose and to benchmark different matrix implementations.

### Features:
- **sm**: float matrix (OpenBLAS or Apple's Accelerator, Apple's MPS)
- **dm**: double matrix (OpenBLAS or Apple's Accelerator)
- **dms**: double sparse matrix in COO format (using SuiteSparse)
- **i/o**: console-print, 
  - read/write MAT files (using libs matio, hdf5)
  - read/wirte MarketMatrix files


### Performance
**sm:**
- SIMD acceleration using OpenMP and NEON intrinsics for `sm` operations (add, diff, multiply, norm, etc.) - if available
- Optimized fallback paths with transposed access to improve cache locality
- Uses Apple Accelerate or OpenBLAS depending on build configuration
- Optional GPU acceleration via Apple Metal Performance Shaders (MPS) for large float matrix multiplications


### Build
To build library. Be sure openmp, openblas, suitesparse and matio is installed locally (via i.e. homebrew). Using [Bazel](https://bazel.build/):

```bash
bazel build //src:matrix --define USE_ACCELERATE=1  
```

### Example


### Tests
```bash
bazel test //tests:all
```

### License:
see License File (MIT)
