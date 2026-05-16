# Tensor NN forward pass example

This example demonstrates a small but realistic neural-network forward pass
using `app/tensor` primitives:

- input tensor creation (`st_create`)
- 2D convolution (`st_conv2d_nchw`)
- fused batch normalization + ReLU (`st_batchnorm2d_forward_relu`)
- global average pooling (`st_global_avgpool2d_nchw`)
- score readout (`st_get`)
- cleanup (`st_destroy`)

## Source

- `doc/examples/tensor_basic_example.c`

## Build and run

- Build: `bazel build //doc/examples:tensor_basic_example`
- Run: `bazel run //doc/examples:tensor_basic_example`

## Expected output (example)

```
NN forward pass complete (Conv -> BatchNorm+ReLU -> GlobalAvgPool)
Output scores [N=1, C=2, H=1, W=1]:
  class_0: ...
  class_1: ...
```
