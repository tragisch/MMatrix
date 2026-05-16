# `st.h` – Public Core Tensor API

Public core API for float tensor creation, views, element-wise ops, and utilities.

## Types

### `StDtype`

Element type used by tensor storage.

Values:

- `ST_DTYPE_F32`
- `ST_DTYPE_BF16`

### `StLayout`

Logical tensor layout hint.

Values:

- `ST_LAYOUT_CONTIGUOUS`
- `ST_LAYOUT_NCHW`
- `ST_LAYOUT_NHWC`
- `ST_LAYOUT_TBF`
- `ST_LAYOUT_BTF`

### Typedefs

- `struct StBuffer`
- `enum StDtype` — Element type used by tensor storage.
- `enum StLayout` — Logical tensor layout hint.
- `struct FloatMatrix`
- `struct FloatTensor` — Opaque tensor handle used by all public tensor APIs.

## Functions

### `st_dtype_size`

`size_t st_dtype_size(StDtype dtype)`

Return storage size in bytes for a tensor element type.

Parameters:

- `dtype` (`StDtype`): Element type.

Returns: 4 for ST_DTYPE_F32, 2 for ST_DTYPE_BF16.

### `st_compute_default_strides`

`bool st_compute_default_strides(size_t ndim, const size_t *shape, ptrdiff_t *out_strides)`

Compute default row-major strides for a shape.

Parameters:

- `ndim` (`size_t`): Number of dimensions.
- `shape` (`const size_t *`): Input shape array of length ndim.
- `out_strides` (`ptrdiff_t *`): Output stride array of length ndim.

### `st_numel_from_shape`

`bool st_numel_from_shape(size_t ndim, const size_t *shape, size_t *out_numel)`

Compute number of elements from a shape.

Parameters:

- `ndim` (`size_t`): Number of dimensions.
- `shape` (`const size_t *`): Input shape array of length ndim.
- `out_numel` (`size_t *`): Output element count.

### `st_create`

`st_create(size_t ndim, const size_t *shape)`

Create a contiguous f32 tensor with zero-initialized storage.

Parameters:

- `ndim` (`size_t`): Number of dimensions.
- `shape` (`const size_t *`): Shape array of length ndim.

Returns: New tensor on success, or NULL on error.

### `st_create_bf16`

`st_create_bf16(size_t ndim, const size_t *shape)`

Create a contiguous bf16 tensor with zero-initialized storage.

Parameters:

- `ndim` (`size_t`): Number of dimensions.
- `shape` (`const size_t *`): Shape array of length ndim.

Returns: New tensor on success, or NULL on error.

### `st_create_with_data`

`st_create_with_data(size_t ndim, const size_t *shape, float *data, size_t capacity, bool take_ownership)`

Wrap existing f32 data in a tensor object.

Parameters:

- `ndim` (`size_t`): Number of dimensions.
- `shape` (`const size_t *`): Shape array of length ndim.
- `data` (`float *`): Pointer to existing f32 storage.
- `capacity` (`size_t`): Capacity in float elements of data.
- `take_ownership` (`bool`): When true, tensor destructor will free data.

Returns: New tensor on success, or NULL on error.

### `st_clone`

`st_clone(const FloatTensor *src)`

Create deep copy of a tensor.

Parameters:

- `src` (`const FloatTensor *`): Source tensor.

Returns: New cloned tensor on success, or NULL on error.

### `st_to_f32`

`st_to_f32(const FloatTensor *src)`

Convert tensor to f32 storage.

Parameters:

- `src` (`const FloatTensor *`): Source tensor.

Returns: New f32 tensor, or NULL on error.

### `st_to_bf16`

`st_to_bf16(const FloatTensor *src)`

Convert tensor to bf16 storage.

Parameters:

- `src` (`const FloatTensor *`): Source tensor.

Returns: New bf16 tensor, or NULL on error.

### `st_view`

`st_view(FloatTensor *base, size_t ndim, const size_t *shape, const ptrdiff_t *strides, size_t offset_elements)`

Create view with custom shape, strides, and offset.

Parameters:

- `base` (`FloatTensor *`): Base tensor to view into.
- `ndim` (`size_t`): Number of dimensions in the view.
- `shape` (`const size_t *`): View shape array.
- `strides` (`const ptrdiff_t *`): View stride array.
- `offset_elements` (`size_t`): Element offset from base start.

Returns: New view tensor, or NULL on error.

### `st_is_contiguous`

`bool st_is_contiguous(const FloatTensor *tensor)`

Check whether tensor has default contiguous row-major strides.

Parameters:

- `tensor` (`const FloatTensor *`): Tensor to inspect.

### `st_reshape`

`bool st_reshape(FloatTensor *tensor, size_t new_ndim, const size_t *new_shape)`

Reshape tensor in-place.

Parameters:

- `tensor` (`FloatTensor *`): Tensor to reshape.
- `new_ndim` (`size_t`): Number of new dimensions.
- `new_shape` (`const size_t *`): New shape array.

### `st_permute_view`

`st_permute_view(FloatTensor *base, const size_t *perm)`

Create a permuted view without copying storage.

Parameters:

- `base` (`FloatTensor *`): Base tensor.
- `perm` (`const size_t *`): Permutation array of length tensor ndim.

Returns: New permuted view, or NULL on error.

### `st_get`

`float st_get(const FloatTensor *tensor, const size_t *indices)`

Read one scalar from a tensor at multi-index.

Parameters:

- `tensor` (`const FloatTensor *`): Source tensor.
- `indices` (`const size_t *`): Multi-index array of length tensor ndim.

Returns: Scalar value, or 0.0f on invalid input.

### `st_set`

`bool st_set(FloatTensor *tensor, const size_t *indices, float value)`

Write one scalar into a tensor at multi-index.

Parameters:

- `tensor` (`FloatTensor *`): Destination tensor.
- `indices` (`const size_t *`): Multi-index array of length tensor ndim.
- `value` (`float`): Scalar value to write.

### `st_as_sm_view`

`bool st_as_sm_view(const FloatTensor *tensor, FloatMatrix *out_view)`

Expose 2D contiguous f32 tensor as FloatMatrix view.

Parameters:

- `tensor` (`const FloatTensor *`): Source tensor.
- `out_view` (`FloatMatrix *`): Output matrix view.

### `st_destroy`

`void st_destroy(FloatTensor *tensor)`

Destroy tensor handle and release associated storage/resources.

Parameters:

- `tensor` (`FloatTensor *`): Tensor to destroy (NULL-safe).

### `st_tensor_sync`

`void st_tensor_sync(FloatTensor *tensor)`

Wait for pending async GPU writes to complete.

Parameters:

- `tensor` (`FloatTensor *`): Tensor to synchronize (NULL-safe).

### `st_tensor_ndim`

`size_t st_tensor_ndim(const FloatTensor *tensor)`

Return number of dimensions, or 0 for NULL tensor.

Parameters:

- `tensor` (`const FloatTensor *`)

### `st_tensor_shape`

`constsize_t* st_tensor_shape(const FloatTensor *tensor)`

Return shape pointer, or NULL for NULL tensor.

Parameters:

- `tensor` (`const FloatTensor *`)

### `st_tensor_numel`

`size_t st_tensor_numel(const FloatTensor *tensor)`

Return logical element count, or 0 for NULL tensor.

Parameters:

- `tensor` (`const FloatTensor *`)

### `st_tensor_dtype`

`st_tensor_dtype(const FloatTensor *tensor)`

Return dtype, defaults to ST_DTYPE_F32 for NULL tensor.

Parameters:

- `tensor` (`const FloatTensor *`)

### `st_tensor_data`

`constfloat* st_tensor_data(const FloatTensor *tensor)`

Return raw const data pointer.

Parameters:

- `tensor` (`const FloatTensor *`)

### `st_tensor_mutable_data`

`float* st_tensor_mutable_data(FloatTensor *tensor)`

Return raw mutable data pointer.

Parameters:

- `tensor` (`FloatTensor *`)

### `st_inplace_add`

`bool st_inplace_add(FloatTensor *a, const FloatTensor *b)`

In-place add: a[i] += b[i].

Parameters:

- `a` (`FloatTensor *`)
- `b` (`const FloatTensor *`)

### `st_inplace_sub`

`bool st_inplace_sub(FloatTensor *a, const FloatTensor *b)`

In-place subtract: a[i] -= b[i].

Parameters:

- `a` (`FloatTensor *`)
- `b` (`const FloatTensor *`)

### `st_inplace_scale`

`bool st_inplace_scale(FloatTensor *t, float scalar)`

In-place scale: t[i] *= scalar.

Parameters:

- `t` (`FloatTensor *`)
- `scalar` (`float`)

### `st_inplace_elementwise_multiply`

`bool st_inplace_elementwise_multiply(FloatTensor *a, const FloatTensor *b)`

In-place Hadamard product: a[i] *= b[i].

Parameters:

- `a` (`FloatTensor *`)
- `b` (`const FloatTensor *`)

### `st_fill`

`bool st_fill(FloatTensor *t, float value)`

Fill all tensor elements with one scalar value.

Parameters:

- `t` (`FloatTensor *`)
- `value` (`float`)

### `st_apply_relu`

`bool st_apply_relu(FloatTensor *t)`

Apply ReLU in-place: t[i] = max(0, t[i]).

Parameters:

- `t` (`FloatTensor *`)

### `st_apply_relu_backward`

`bool st_apply_relu_backward(const FloatTensor *activation, FloatTensor *grad)`

Apply ReLU backward mask in-place to grad.

Parameters:

- `activation` (`const FloatTensor *`)
- `grad` (`FloatTensor *`)

### `st_sum_axes`

`st_sum_axes(const FloatTensor *t, const size_t *axes, size_t num_axes)`

Reduce tensor by summing over selected axes.

Parameters:

- `t` (`const FloatTensor *`): Input tensor.
- `axes` (`const size_t *`): Array of axis indices to reduce.
- `num_axes` (`size_t`): Number of axes in axes.

Returns: Reduced tensor on success, or NULL on error.

### `st_pad_nchw`

`st_pad_nchw(const FloatTensor *input, size_t pad_h, size_t pad_w, float value)`

Pad NCHW tensor spatially.

Parameters:

- `input` (`const FloatTensor *`): Input tensor with shape [N, C, H, W].
- `pad_h` (`size_t`): Symmetric padding in height dimension.
- `pad_w` (`size_t`): Symmetric padding in width dimension.
- `value` (`float`): Padding constant.

Returns: New padded tensor on success, or NULL on error.

### `st_mps_warmup_shapes`

`void st_mps_warmup_shapes(const StWarmupShape *shapes, size_t count)`

Pre-compile MPS graphs for a list of tensor shapes.

Parameters:

- `shapes` (`const StWarmupShape *`): Array of warmup descriptors.
- `count` (`size_t`): Number of entries in shapes.
