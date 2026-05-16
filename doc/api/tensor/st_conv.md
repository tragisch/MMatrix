# `st_conv.h` – Convolution API

Public API for 2D convolution and fused conv blocks in NCHW layout.

## Types

### `StConvBackend`

Backend selector for convolution execution.

Values:

- `ST_CONV_BACKEND_AUTO`
- `ST_CONV_BACKEND_REFERENCE`
- `ST_CONV_BACKEND_CPU_OPT`
- `ST_CONV_BACKEND_GEMM`
- `ST_CONV_BACKEND_MPS`
- `ST_CONV_BACKEND_BNNS`

### `StPoolType`

Pooling operator used in fused conv+bn+pool execution.

Values:

- `ST_POOL_MAX`
- `ST_POOL_AVG`

### Typedefs

- `enum StConvBackend` — Backend selector for convolution execution.
- `struct StConv2dParams StConv2dParams` — Parameter bundle for 2D convolution in NCHW layout.
- `enum StPoolType` — Pooling operator used in fused conv+bn+pool execution.
- `struct StPool2dParams StPool2dParams` — Parameter bundle for 2D pooling in NCHW layout.

## Functions

### `st_conv2d_backward_data_nchw`

`bool st_conv2d_backward_data_nchw(const FloatTensor *grad_output, const FloatTensor *weight, const StConv2dParams *params, FloatTensor *grad_input)`

Compute gradient with respect to convolution input.

Parameters:

- `grad_output` (`const FloatTensor *`)
- `weight` (`const FloatTensor *`)
- `params` (`const StConv2dParams *`)
- `grad_input` (`FloatTensor *`)

### `st_conv2d_backward_weight_nchw`

`bool st_conv2d_backward_weight_nchw(const FloatTensor *input, const FloatTensor *grad_output, const StConv2dParams *params, FloatTensor *grad_weight)`

Compute gradient with respect to convolution weights.

Parameters:

- `input` (`const FloatTensor *`)
- `grad_output` (`const FloatTensor *`)
- `params` (`const StConv2dParams *`)
- `grad_weight` (`FloatTensor *`)

### `st_conv2d_backward_bias`

`bool st_conv2d_backward_bias(const FloatTensor *grad_output, FloatTensor *grad_bias)`

Compute gradient with respect to convolution bias.

Parameters:

- `grad_output` (`const FloatTensor *`)
- `grad_bias` (`FloatTensor *`)

### `st_conv2d_default_params`

`st_conv2d_default_params(void)`

Return default convolution parameters.

Parameters:

- `(unnamed)` (`void`)

Returns: Parameters with stride=1, pad=0, dilation=1, backend=AUTO.

### `st_conv2d_output_hw`

`bool st_conv2d_output_hw(size_t in_h, size_t in_w, size_t kernel_h, size_t kernel_w, const StConv2dParams *params, size_t *out_h, size_t *out_w)`

Compute NCHW output spatial size for a 2D convolution.

Parameters:

- `in_h` (`size_t`): Input height.
- `in_w` (`size_t`): Input width.
- `kernel_h` (`size_t`): Kernel height.
- `kernel_w` (`size_t`): Kernel width.
- `params` (`const StConv2dParams *`): Convolution parameters.
- `out_h` (`size_t *`): Output height.
- `out_w` (`size_t *`): Output width.

### `st_conv2d_nchw`

`bool st_conv2d_nchw(const FloatTensor *input, const FloatTensor *weight, const FloatTensor *bias, const StConv2dParams *params, FloatTensor *output)`

Execute 2D convolution for NCHW tensors.

Parameters:

- `input` (`const FloatTensor *`): Input tensor [N, Cin, H, W].
- `weight` (`const FloatTensor *`): Weight tensor [Cout, Cin, Kh, Kw].
- `bias` (`const FloatTensor *`): Optional bias tensor [Cout] (NULL allowed).
- `params` (`const StConv2dParams *`): Convolution parameters.
- `output` (`FloatTensor *`): Output tensor [N, Cout, outH, outW].

### `st_conv2d_batchnorm2d_forward_nchw`

`bool st_conv2d_batchnorm2d_forward_nchw(const FloatTensor *input, const FloatTensor *weight, const FloatTensor *bias, const StConv2dParams *params, const FloatTensor *gamma, const FloatTensor *beta, float epsilon, FloatTensor *output, FloatTensor *mean, FloatTensor *var, bool apply_relu)`

Execute fused Conv2D + BatchNorm2D forward.

Parameters:

- `input` (`const FloatTensor *`): Input tensor [N, Cin, H, W].
- `weight` (`const FloatTensor *`): Weight tensor [Cout, Cin, Kh, Kw].
- `bias` (`const FloatTensor *`): Optional bias tensor [Cout].
- `params` (`const StConv2dParams *`): Convolution parameters.
- `gamma` (`const FloatTensor *`): BatchNorm scale [Cout] (NULL means identity).
- `beta` (`const FloatTensor *`): BatchNorm shift [Cout] (NULL means zero).
- `epsilon` (`float`): Numerical stability constant.
- `output` (`FloatTensor *`): Output tensor.
- `mean` (`FloatTensor *`): Optional channel means [Cout].
- `var` (`FloatTensor *`): Optional channel variances [Cout].
- `apply_relu` (`bool`): Apply ReLU after BatchNorm.

### `st_conv2d_batchnorm2d_pool_forward_nchw`

`bool st_conv2d_batchnorm2d_pool_forward_nchw(const FloatTensor *input, const FloatTensor *weight, const FloatTensor *bias, const StConv2dParams *conv_params, const FloatTensor *gamma, const FloatTensor *beta, float epsilon, const StPool2dParams *pool_params, FloatTensor *output, FloatTensor *mean, FloatTensor *var, bool apply_relu)`

Execute fused Conv2D + BatchNorm2D + Pool2D forward.

Parameters:

- `input` (`const FloatTensor *`): Input tensor.
- `weight` (`const FloatTensor *`): Weight tensor.
- `bias` (`const FloatTensor *`): Optional bias tensor.
- `conv_params` (`const StConv2dParams *`): Convolution parameters.
- `gamma` (`const FloatTensor *`): BatchNorm scale tensor.
- `beta` (`const FloatTensor *`): BatchNorm shift tensor.
- `epsilon` (`float`): Numerical stability constant.
- `pool_params` (`const StPool2dParams *`): Pooling parameters.
- `output` (`FloatTensor *`): Output tensor.
- `mean` (`FloatTensor *`): Optional channel means tensor.
- `var` (`FloatTensor *`): Optional channel variances tensor.
- `apply_relu` (`bool`): Apply ReLU between BatchNorm and Pool.

### `st_conv_set_mps_thresholds`

`bool st_conv_set_mps_thresholds(double macs_threshold, size_t out_elems_threshold)`

Override MPS AUTO dispatch thresholds for standalone convolution.

Parameters:

- `macs_threshold` (`double`): Minimum MACs to use MPS in AUTO mode.
- `out_elems_threshold` (`size_t`): Minimum output element count to use MPS in AUTO mode.

### `st_conv_get_mps_thresholds`

`void st_conv_get_mps_thresholds(double *out_macs_threshold, size_t *out_out_elems_threshold)`

Query current MPS AUTO dispatch thresholds.

Parameters:

- `out_macs_threshold` (`double *`): Output MAC threshold (nullable).
- `out_out_elems_threshold` (`size_t *`): Output element threshold (nullable).

### `st_conv_reload_mps_thresholds_from_env`

`void st_conv_reload_mps_thresholds_from_env(void)`

Reload MPS AUTO thresholds from environment variables.

Parameters:

- `(unnamed)` (`void`)

### `st_conv2d_last_backend`

`constchar* st_conv2d_last_backend(void)`

Return backend label used by the latest st_conv2d_nchw call.

Parameters:

- `(unnamed)` (`void`)

Returns: Backend name string (never NULL).
