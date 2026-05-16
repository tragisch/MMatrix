# `st.h` – Public Core Tensor API

Public core API for float tensor creation, views, element-wise ops, and utilities.

This header intentionally exports the opaque `FloatTensor` type for external use.

## Types

### `StDtype`

- `ST_DTYPE_F32` (IEEE 754 binary32, 4 bytes)
- `ST_DTYPE_BF16` (bfloat16, 2 bytes)

Helper:

- `st_dtype_size(dtype)` → 4 (`f32`) or 2 (`bf16`)

### `StLayout`

- `ST_LAYOUT_CONTIGUOUS`
- `ST_LAYOUT_NCHW`
- `ST_LAYOUT_NHWC`
- `ST_LAYOUT_TBF`
- `ST_LAYOUT_BTF`

### `StWarmupShape`

Warmup descriptor for shape-aware MPSGraph precompilation.

## Core functions

### Creation / conversion / views

- `st_create`, `st_create_bf16`, `st_create_with_data`
- `st_clone`, `st_to_f32`, `st_to_bf16`
- `st_view`, `st_permute_view`, `st_reshape`

### Metadata / access

- `st_tensor_ndim`, `st_tensor_shape`, `st_tensor_numel`, `st_tensor_dtype`
- `st_tensor_data`, `st_tensor_mutable_data`
- `st_get`, `st_set`
- `st_is_contiguous`, `st_as_sm_view`

### In-place operations

- `st_inplace_add`, `st_inplace_sub`, `st_inplace_scale`
- `st_inplace_elementwise_multiply`
- `st_fill`

### Activation / reduction / padding

- `st_apply_relu`, `st_apply_relu_backward`
- `st_sum_axes`
- `st_pad_nchw`

### Lifecycle / synchronization

- `st_destroy`
- `st_tensor_sync`

### Shape and utility helpers

- `st_compute_default_strides`
- `st_numel_from_shape`

### MPS-Warmup

- `st_mps_warmup_shapes`

Warms up Conv2D (when `c_out > 0`), MaxPool2D, AvgPool2D, and BatchNorm2D.
