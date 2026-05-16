# `st_batchnorm.h` – BatchNorm API

Public API for BatchNorm2D and fused BatchNorm+ReLU in NCHW layout.

## Functions

### `st_batchnorm2d_forward`

Forward BatchNorm2D over the channel dimension for NCHW tensors.

- `input`: input tensor `[N, C, H, W]`
- `gamma`: optional scale tensor `[C]` (`NULL` = ones)
- `beta`: optional shift tensor `[C]` (`NULL` = zeros)
- `epsilon`: numerical stability constant
- `output`: output tensor `[N, C, H, W]`
- `mean`: optional output `[C]` (channel means)
- `var`: optional output `[C]` (channel variances)

Returns `true` on success, otherwise `false`.

### `st_batchnorm2d_backward`

Backward pass of BatchNorm2D for NCHW tensors.

- `grad_output`: upstream gradient `[N, C, H, W]`
- `input`: original forward input `[N, C, H, W]`
- `mean`: channel means from forward pass `[C]`
- `var`: channel variances from forward pass `[C]`
- `gamma`: optional scale tensor `[C]`
- `epsilon`: same `epsilon` value used in forward pass
- `grad_input`: gradient w.r.t. input `[N, C, H, W]`
- `grad_gamma`: optional gradient w.r.t. `gamma` `[C]`
- `grad_beta`: optional gradient w.r.t. `beta` `[C]`

Returns `true` on success, otherwise `false`.

### `st_batchnorm2d_forward_relu`

Forward pass for fused BatchNorm2D + ReLU.

- `input`: input tensor `[N, C, H, W]`
- `gamma`: optional scale tensor `[C]`
- `beta`: optional shift tensor `[C]`
- `epsilon`: numerical stability constant
- `output`: output tensor `[N, C, H, W]` after ReLU
- `mean`: optional channel mean `[C]`
- `var`: optional channel variance `[C]`

Returns `true` on success, otherwise `false`.

### `st_batchnorm2d_backward_relu`

Backward pass for fused BatchNorm2D + ReLU.

- `grad_output`: upstream gradient
- `input`: original forward input
- `bn_output`: output of `st_batchnorm2d_forward_relu` (post-ReLU)
- `mean`: channel means from forward pass
- `var`: channel variances from forward pass
- `gamma`: optional scale tensor
- `epsilon`: same `epsilon` value used in forward pass
- `grad_input`: gradient w.r.t. input
- `grad_gamma`: optional gradient w.r.t. `gamma`
- `grad_beta`: optional gradient w.r.t. `beta`

Returns `true` on success, otherwise `false`.
