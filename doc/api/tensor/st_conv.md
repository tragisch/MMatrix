# `st_conv.h` – Convolution API

Public API for 2D convolution and fused conv blocks in NCHW layout.

## Types

### `StConvBackend`

Backend selector for convolution execution.

- `ST_CONV_BACKEND_AUTO`
- `ST_CONV_BACKEND_REFERENCE`
- `ST_CONV_BACKEND_CPU_OPT`
- `ST_CONV_BACKEND_GEMM`
- `ST_CONV_BACKEND_MPS`
- `ST_CONV_BACKEND_BNNS`

### `StConv2dParams`

Parameter bundle for 2D convolution in NCHW:

- `stride_h`, `stride_w`
- `pad_h`, `pad_w`
- `dilation_h`, `dilation_w`
- `backend`

### `StPoolType`

Pooling operator for fused Conv+BN+Pool:

- `ST_POOL_MAX`
- `ST_POOL_AVG`

### `StPool2dParams`

Pooling parameters in NCHW:

- `pool_type`
- `kernel_h`, `kernel_w`
- `stride_h`, `stride_w`
- `pad_h`, `pad_w`

## Functions

### `st_conv2d_default_params`

Returns default parameters (`stride=1`, `pad=0`, `dilation=1`, `backend=AUTO`).

### `st_conv2d_output_hw`

Computes NCHW output size for a 2D convolution.

- Input: `in_h`, `in_w`, `kernel_h`, `kernel_w`, `params`
- Output: `out_h`, `out_w`

Returns `true` on success, otherwise `false` (including overflow/invalid params).

### `st_conv2d_nchw`

Executes 2D convolution for NCHW tensors.

- `input`: `[N, Cin, H, W]`
- `weight`: `[Cout, Cin, Kh, Kw]`
- `bias`: optional `[Cout]`
- `params`: convolution parameters
- `output`: `[N, Cout, outH, outW]`

### `st_conv2d_batchnorm2d_forward_nchw`

Fused Conv2D + BatchNorm2D forward.

- In addition to conv inputs: `gamma`, `beta`, `epsilon`, `mean`, `var`, `apply_relu`

### `st_conv2d_batchnorm2d_pool_forward_nchw`

Fused Conv2D + BatchNorm2D + Pool2D forward.

- Additional parameter: `pool_params`

### MPS auto-dispatch controls

- `st_conv_set_mps_thresholds`
- `st_conv_get_mps_thresholds`
- `st_conv_reload_mps_thresholds_from_env`
- `st_conv2d_last_backend`

### Backward passes

- `st_conv2d_backward_data_nchw`: gradient w.r.t. input
- `st_conv2d_backward_weight_nchw`: gradient w.r.t. weights
- `st_conv2d_backward_bias`: gradient w.r.t. bias
