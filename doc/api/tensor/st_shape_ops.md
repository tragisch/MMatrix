# `st_shape_ops.h` – Shape/View Operations API

Public API for tensor shape/view transformation operations.

## Functions

### `st_flatten`

`st_flatten(const FloatTensor *tensor, size_t start_axis, size_t end_axis)`

Flatten a range of axes into a single axis.

Parameters:

- `tensor` (`const FloatTensor *`): Input tensor.
- `start_axis` (`size_t`): First axis to merge (inclusive).
- `end_axis` (`size_t`): Last axis to merge (exclusive).

Returns: New flattened view (no data copy), or NULL on error.

### `st_flatten_all`

`st_flatten_all(const FloatTensor *tensor)`

Flatten all axes into a 1D view.

Parameters:

- `tensor` (`const FloatTensor *`): Input tensor.

Returns: New 1D view (no data copy), or NULL on error.

### `st_permute`

`st_permute(const FloatTensor *tensor, const size_t *axes)`

Permute tensor axes.

Parameters:

- `tensor` (`const FloatTensor *`): Input tensor.
- `axes` (`const size_t *`): Permutation array where axes[i] maps output axis i.

Returns: New permuted view (no data copy), or NULL on error.

### `st_reshape_to`

`bool st_reshape_to(FloatTensor *tensor, size_t new_ndim, const size_t *new_shape)`

Convenience alias for st_reshape.

Parameters:

- `tensor` (`FloatTensor *`)
- `new_ndim` (`size_t`)
- `new_shape` (`const size_t *`)

### `st_concat`

`st_concat(const FloatTensor *const *tensors, size_t num_tensors, size_t axis)`

Concatenate multiple tensors along one axis.

Parameters:

- `tensors` (`const FloatTensor *const *`): Array of input tensors.
- `num_tensors` (`size_t`): Number of entries in tensors.
- `axis` (`size_t`): Concatenation axis.

Returns: New contiguous tensor on success, or NULL on error.

### `st_squeeze`

`st_squeeze(const FloatTensor *tensor)`

Remove all dimensions of size 1.

Parameters:

- `tensor` (`const FloatTensor *`): Input tensor.

Returns: New squeezed view (no data copy), or NULL on error.

### `st_squeeze_dim`

`st_squeeze_dim(const FloatTensor *tensor, size_t axis)`

Remove one dimension if and only if its size is 1.

Parameters:

- `tensor` (`const FloatTensor *`): Input tensor.
- `axis` (`size_t`): Axis to remove.

Returns: New squeezed view (no data copy), or NULL on error.

### `st_unsqueeze`

`st_unsqueeze(const FloatTensor *tensor, size_t axis)`

Insert a new dimension of size 1.

Parameters:

- `tensor` (`const FloatTensor *`): Input tensor.
- `axis` (`size_t`): Insert position in [0, ndim].

Returns: New unsqueezed view (no data copy), or NULL on error.

### `st_expand`

`st_expand(const FloatTensor *tensor, size_t axis, size_t count)`

Expand a size-1 dimension by repetition.

Parameters:

- `tensor` (`const FloatTensor *`): Input tensor.
- `axis` (`size_t`): Axis to expand (must currently be size 1).
- `count` (`size_t`): Replication count.

Returns: New tensor with copied data, or NULL on error.

### `st_split`

`bool st_split(const FloatTensor *tensor, size_t axis, size_t num_splits, FloatTensor ***out_splits)`

Split a tensor into equal parts along one axis.

Parameters:

- `tensor` (`const FloatTensor *`): Input tensor.
- `axis` (`size_t`): Split axis.
- `num_splits` (`size_t`): Number of equally sized splits.
- `out_splits` (`FloatTensor ***`): Output array of split tensor handles.
