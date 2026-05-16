# `st_pool.h` – Pooling API

Public API for pooling operators in NCHW layout.

## Functions

### `st_pool2d_output_hw`

Computes 2D pooling output size.

- Input: `in_h`, `in_w`, `kernel_h`, `kernel_w`, `stride_h`, `stride_w`, `pad_h`, `pad_w`
- Output: `out_h`, `out_w`

Returns `true` on success, otherwise `false`.

### `st_maxpool2d_nchw`

Forward max pooling for NCHW tensors.

- `input`: `[N, C, H, W]`
- `output`: `[N, C, outH, outW]`
- `indices`: optional `[N, C, outH, outW]` with max indices

### `st_avgpool2d_nchw`

Forward average pooling for NCHW tensors.

- `input`: `[N, C, H, W]`
- `output`: `[N, C, outH, outW]`

### `st_maxpool2d_backward_nchw`

Backward max pooling for NCHW tensors.

- `grad_output`: `[N, C, outH, outW]`
- `indices`: saved max indices from forward pass
- `input_h`, `input_w`: original input size
- `grad_input`: `[N, C, H, W]` (written/accumulated)

### `st_avgpool2d_backward_nchw`

Backward average pooling for NCHW tensors.

- `grad_output`: `[N, C, outH, outW]`
- window/stride/padding parameters equivalent to forward pass
- `grad_input`: `[N, C, H, W]` (written/accumulated)

### `st_global_avgpool2d_nchw`

Global average pooling over spatial dimensions.

- `input`: `[N, C, H, W]`
- `output`: `[N, C, 1, 1]`

### `st_global_avgpool2d_backward_nchw`

Backward global average pooling.

- `grad_output`: `[N, C, 1, 1]`
- `grad_input`: `[N, C, H, W]`
