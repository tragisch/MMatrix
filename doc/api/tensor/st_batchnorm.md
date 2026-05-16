# `st_batchnorm.h` – BatchNorm API

Public API for BatchNorm2D and fused BatchNorm+ReLU in NCHW layout.

## Functions

### `st_batchnorm2d_forward`

`bool st_batchnorm2d_forward(const FloatTensor *input, const FloatTensor *gamma, const FloatTensor *beta, float epsilon, FloatTensor *output, FloatTensor *mean, FloatTensor *var)`

Forward BatchNorm2D over channel dimension for NCHW tensors.

Parameters:

- `input` (`const FloatTensor *`): Input tensor [N, C, H, W].
- `gamma` (`const FloatTensor *`): Optional scale tensor [C] (NULL means all ones).
- `beta` (`const FloatTensor *`): Optional shift tensor [C] (NULL means all zeros).
- `epsilon` (`float`): Numerical stability constant.
- `output` (`FloatTensor *`): Output tensor [N, C, H, W].
- `mean` (`FloatTensor *`): Optional output tensor [C] for channel means.
- `var` (`FloatTensor *`): Optional output tensor [C] for channel variances.

### `st_batchnorm2d_backward`

`bool st_batchnorm2d_backward(const FloatTensor *grad_output, const FloatTensor *input, const FloatTensor *mean, const FloatTensor *var, const FloatTensor *gamma, float epsilon, FloatTensor *grad_input, FloatTensor *grad_gamma, FloatTensor *grad_beta)`

Backward BatchNorm2D for NCHW tensors.

Parameters:

- `grad_output` (`const FloatTensor *`): Upstream gradient [N, C, H, W].
- `input` (`const FloatTensor *`): Original forward input [N, C, H, W].
- `mean` (`const FloatTensor *`): Channel means from forward pass [C].
- `var` (`const FloatTensor *`): Channel variances from forward pass [C].
- `gamma` (`const FloatTensor *`): Optional scale tensor [C].
- `epsilon` (`float`): Same epsilon used in forward pass.
- `grad_input` (`FloatTensor *`): Gradient w.r.t. input [N, C, H, W].
- `grad_gamma` (`FloatTensor *`): Optional gradient w.r.t. gamma [C].
- `grad_beta` (`FloatTensor *`): Optional gradient w.r.t. beta [C].

### `st_batchnorm2d_forward_relu`

`bool st_batchnorm2d_forward_relu(const FloatTensor *input, const FloatTensor *gamma, const FloatTensor *beta, float epsilon, FloatTensor *output, FloatTensor *mean, FloatTensor *var)`

Forward fused BatchNorm2D + ReLU.

Parameters:

- `input` (`const FloatTensor *`): Input tensor [N, C, H, W].
- `gamma` (`const FloatTensor *`): Optional scale tensor [C].
- `beta` (`const FloatTensor *`): Optional shift tensor [C].
- `epsilon` (`float`): Numerical stability constant.
- `output` (`FloatTensor *`): Output tensor [N, C, H, W] after ReLU.
- `mean` (`FloatTensor *`): Optional channel means [C].
- `var` (`FloatTensor *`): Optional channel variances [C].

### `st_batchnorm2d_backward_relu`

`bool st_batchnorm2d_backward_relu(const FloatTensor *grad_output, const FloatTensor *input, const FloatTensor *bn_output, const FloatTensor *mean, const FloatTensor *var, const FloatTensor *gamma, float epsilon, FloatTensor *grad_input, FloatTensor *grad_gamma, FloatTensor *grad_beta)`

Backward fused BatchNorm2D + ReLU.

Parameters:

- `grad_output` (`const FloatTensor *`): Upstream gradient.
- `input` (`const FloatTensor *`): Original forward input.
- `bn_output` (`const FloatTensor *`): Output of st_batchnorm2d_forward_relu (post-ReLU).
- `mean` (`const FloatTensor *`): Channel means from forward pass.
- `var` (`const FloatTensor *`): Channel variances from forward pass.
- `gamma` (`const FloatTensor *`): Optional scale tensor.
- `epsilon` (`float`): Same epsilon used in forward pass.
- `grad_input` (`FloatTensor *`): Gradient w.r.t. input.
- `grad_gamma` (`FloatTensor *`): Optional gradient w.r.t. gamma.
- `grad_beta` (`FloatTensor *`): Optional gradient w.r.t. beta.
