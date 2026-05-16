# MMatrix Documentation

This documentation is fully located under `doc/`.

## Tensor API

- [Tensor Core](api/tensor/st.md)
- [BatchNorm](api/tensor/st_batchnorm.md)
- [Convolution](api/tensor/st_conv.md)
- [Pooling](api/tensor/st_pool.md)
- [Shape Ops](api/tensor/st_shape_ops.md)

## Examples

- [Tensor NN forward pass](examples/tensor_basic.md)

Run the example directly:

- `bazel run //doc/examples:tensor_basic_example`

## Quick start

- Build static site: `bazel build //:docs`
- Run local preview with live reload: `bazel run //:docs_serve`
