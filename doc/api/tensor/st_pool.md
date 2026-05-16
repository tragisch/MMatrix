# `st_pool.h` – Pooling API

Public API for pooling operators in NCHW layout.

## Functions

### `st_pool2d_output_hw`

`bool st_pool2d_output_hw(size_t in_h, size_t in_w, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, size_t *out_h, size_t *out_w)`

Compute 2D pooling output size.

Parameters:

- `in_h` (`size_t`): Input height.
- `in_w` (`size_t`): Input width.
- `kernel_h` (`size_t`): Kernel height.
- `kernel_w` (`size_t`): Kernel width.
- `stride_h` (`size_t`): Vertical stride.
- `stride_w` (`size_t`): Horizontal stride.
- `pad_h` (`size_t`): Vertical symmetric padding.
- `pad_w` (`size_t`): Horizontal symmetric padding.
- `out_h` (`size_t *`): Output height.
- `out_w` (`size_t *`): Output width.

### `st_maxpool2d_nchw`

`bool st_maxpool2d_nchw(const FloatTensor *input, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, FloatTensor *output, FloatTensor *indices)`

Forward max pooling for NCHW tensors.

Parameters:

- `input` (`const FloatTensor *`): Input tensor [N, C, H, W].
- `kernel_h` (`size_t`): Kernel height.
- `kernel_w` (`size_t`): Kernel width.
- `stride_h` (`size_t`): Vertical stride.
- `stride_w` (`size_t`): Horizontal stride.
- `pad_h` (`size_t`): Vertical symmetric padding.
- `pad_w` (`size_t`): Horizontal symmetric padding.
- `output` (`FloatTensor *`): Output tensor [N, C, outH, outW].
- `indices` (`FloatTensor *`): Optional tensor [N, C, outH, outW] storing max indices.

### `st_avgpool2d_nchw`

`bool st_avgpool2d_nchw(const FloatTensor *input, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, FloatTensor *output)`

Forward average pooling for NCHW tensors.

Parameters:

- `input` (`const FloatTensor *`): Input tensor [N, C, H, W].
- `kernel_h` (`size_t`): Kernel height.
- `kernel_w` (`size_t`): Kernel width.
- `stride_h` (`size_t`): Vertical stride.
- `stride_w` (`size_t`): Horizontal stride.
- `pad_h` (`size_t`): Vertical symmetric padding.
- `pad_w` (`size_t`): Horizontal symmetric padding.
- `output` (`FloatTensor *`): Output tensor [N, C, outH, outW].

### `st_maxpool2d_backward_nchw`

`bool st_maxpool2d_backward_nchw(const FloatTensor *grad_output, const FloatTensor *indices, size_t input_h, size_t input_w, FloatTensor *grad_input)`

Backward max pooling for NCHW tensors.

Parameters:

- `grad_output` (`const FloatTensor *`): Upstream gradient [N, C, outH, outW].
- `indices` (`const FloatTensor *`): Saved max indices from forward pass.
- `input_h` (`size_t`): Original input height.
- `input_w` (`size_t`): Original input width.
- `grad_input` (`FloatTensor *`): Output gradient buffer [N, C, H, W] (written/accumulated).

### `st_avgpool2d_backward_nchw`

`bool st_avgpool2d_backward_nchw(const FloatTensor *grad_output, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w, size_t pad_h, size_t pad_w, FloatTensor *grad_input)`

Backward average pooling for NCHW tensors.

Parameters:

- `grad_output` (`const FloatTensor *`): Upstream gradient [N, C, outH, outW].
- `kernel_h` (`size_t`): Kernel height.
- `kernel_w` (`size_t`): Kernel width.
- `stride_h` (`size_t`): Vertical stride.
- `stride_w` (`size_t`): Horizontal stride.
- `pad_h` (`size_t`): Vertical symmetric padding.
- `pad_w` (`size_t`): Horizontal symmetric padding.
- `grad_input` (`FloatTensor *`): Output gradient buffer [N, C, H, W] (written/accumulated).

### `st_global_avgpool2d_nchw`

`bool st_global_avgpool2d_nchw(const FloatTensor *input, FloatTensor *output)`

Global average pooling over spatial dimensions.

Parameters:

- `input` (`const FloatTensor *`): Input tensor [N, C, H, W].
- `output` (`FloatTensor *`): Output tensor [N, C, 1, 1].

### `st_global_avgpool2d_backward_nchw`

`bool st_global_avgpool2d_backward_nchw(const FloatTensor *grad_output, FloatTensor *grad_input)`

Backward global average pooling.

Parameters:

- `grad_output` (`const FloatTensor *`): Upstream gradient [N, C, 1, 1].
- `grad_input` (`FloatTensor *`): Output gradient [N, C, H, W].
