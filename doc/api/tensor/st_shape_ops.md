# `st_shape_ops.h` – Shape/View Operations API

Public API for tensor shape/view transformation operations.

## Functions

### `st_flatten`

Flattens an axis range into a single axis.

- `start_axis`: inclusive
- `end_axis`: exclusive

Returns a new view (no data copy), or `NULL` on error.

### `st_flatten_all`

Flattens all axes into a 1D view.

### `st_permute`

Permutes tensor axes.

- `axes[i]` defines which input axis maps to output axis `i`.

### `st_reshape_to`

Inline convenience alias for `st_reshape`.

### `st_concat`

Concatenates multiple tensors along one axis.

- `tensors`: array of input tensors
- `num_tensors`: number of tensors
- `axis`: concatenation axis

Returns a new contiguous tensor, or `NULL`.

### `st_concat_varargs`

Macro helper that simplifies `st_concat` usage with array literals.

### `st_squeeze`

Removes all dimensions of size `1`.

### `st_squeeze_dim`

Removes one dimension only if its size is `1`.

### `st_unsqueeze`

Inserts a new dimension of size `1` at position `axis`.

### `st_expand`

Expands a size-1 dimension by repetition.

- `axis` must currently have size `1`.
- `count` is the replication count.

Returns a new tensor with copied data, or `NULL`.

### `st_split`

Splits a tensor into equal parts along one axis.

- `num_splits`: number of equal splits
- `out_splits`: output array with split tensor handles

Returns `true` on success, `false` on error/uneven split.

Note: The caller must destroy each split tensor and then free `*out_splits`.
