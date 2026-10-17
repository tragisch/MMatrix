# `st_interp.h` – Graph Interpreter Kernel API

Public API for graph-interpreter tensor kernels.

## Types

### `StBinOp`

Element-wise binary operations for the broadcast kernels.

Values:

- `ST_BIN_ADD`
- `ST_BIN_SUB`
- `ST_BIN_MUL`
- `ST_BIN_DIV`

### Typedefs

- `enum StBinOp` — Element-wise binary operations for the broadcast kernels.

## Functions

### `st_binary_broadcast`

`st_binary_broadcast(StBinOp op, const FloatTensor *a, const FloatTensor *b)`

Apply a binary op with NumPy-style broadcasting over any axes.

Parameters:

- `op` (`StBinOp`): Operation to apply.
- `a` (`const FloatTensor *`): Left input tensor (contiguous, f32 or bf16).
- `b` (`const FloatTensor *`): Right input tensor (contiguous, f32 or bf16).

Returns: New contiguous f32 tensor on success, or NULL on invalid input, non-broadcastable shapes, or allocation failure.

### `st_binary_broadcast_into`

`bool st_binary_broadcast_into(StBinOp op, const FloatTensor *a, const FloatTensor *b, FloatTensor *out)`

Apply a binary op with broadcasting into a prepared output tensor.

Parameters:

- `op` (`StBinOp`): Operation to apply.
- `a` (`const FloatTensor *`): Left input tensor (contiguous, f32 or bf16).
- `b` (`const FloatTensor *`): Right input tensor (contiguous, f32 or bf16).
- `out` (`FloatTensor *`): Output tensor; contiguous with the broadcast shape of a and b, f32 or bf16. May alias a or b when that operand's shape equals the output shape (the row kernels read and write the same element-wise position, so in-place accumulation like a += bias is safe); it must not partially overlap any operand with a different shape.

### `st_layernorm`

`bool st_layernorm(const FloatTensor *x, const FloatTensor *gamma, const FloatTensor *beta, size_t norm_rank, float eps, FloatTensor *out)`

Layer-normalize over the last norm_rank axes.

Parameters:

- `x` (`const FloatTensor *`): Input tensor (contiguous, f32 or bf16).
- `gamma` (`const FloatTensor *`): Optional per-element scale of the normalized group; NULL or a contiguous tensor with group elements (f32 or bf16).
- `beta` (`const FloatTensor *`): Optional per-element shift; NULL or like gamma.
- `norm_rank` (`size_t`): Number of trailing axes to normalize over, [1, ndim].
- `eps` (`float`): Variance epsilon added before the inverse square root.
- `out` (`FloatTensor *`): Output tensor, contiguous with the same shape as x; f32 or bf16. May alias x (in-place normalization per group).

### `st_reduce_max`

`st_reduce_max(const FloatTensor *x, uint32_t axes_mask, bool keepdims)`

Maximum over the axes selected in axes_mask.

Parameters:

- `x` (`const FloatTensor *`): Input tensor (contiguous, f32 or bf16).
- `axes_mask` (`uint32_t`): Bitmask; axis i is reduced when bit i is set. Bits at or above the tensor rank are rejected.
- `keepdims` (`bool`): Keep reduced axes as size-1 axes in the output.

Returns: New contiguous f32 tensor on success, or NULL on error.

### `st_reduce_max_into`

`bool st_reduce_max_into(const FloatTensor *x, uint32_t axes_mask, bool keepdims, FloatTensor *out)`

Maximum over selected axes into a prepared output tensor.

Parameters:

- `x` (`const FloatTensor *`): Input tensor (contiguous, f32 or bf16).
- `axes_mask` (`uint32_t`): Bitmask of axes to reduce; see st_reduce_max.
- `keepdims` (`bool`): Keep reduced axes as size-1 axes in the output.
- `out` (`FloatTensor *`): Output tensor with the reduced shape, f32 or bf16. Must not alias x (an empty mask copies; use st_reduce_max instead for in-place results).

### `st_matmul_batched`

`bool st_matmul_batched(const FloatTensor *a, const FloatTensor *b, FloatTensor *c)`

Batched matrix product with broadcast over the leading axes.

Parameters:

- `a` (`const FloatTensor *`): Left tensor, rank 2 to 4, contiguous, f32 or bf16; the last two axes are [M, K].
- `b` (`const FloatTensor *`): Right tensor, rank 2 to 4, contiguous, f32 or bf16; the last two axes are [K, N].
- `c` (`FloatTensor *`): Output tensor with the broadcast batch shape plus [M, N], contiguous, f32 or bf16. Must not alias a or b (the GEMM backend reads inputs while accumulating).

### `st_gather`

`bool st_gather(const FloatTensor *table, const int64_t *indices, size_t count, size_t axis, FloatTensor *out)`

Gather slices along one axis, indexed by int64 values.

Parameters:

- `table` (`const FloatTensor *`): Source tensor, contiguous, f32 or bf16.
- `indices` (`const int64_t *`): Index array of length count; negative indices count from the end of axis.
- `count` (`size_t`): Number of indices; must be greater than zero.
- `axis` (`size_t`): Gather axis, below the tensor rank.
- `out` (`FloatTensor *`): Output tensor, contiguous, same dtype as table, shape equal to the table shape with axis replaced by count. Must not alias table.

### `st_gather_bytes`

`bool st_gather_bytes(const void *table, size_t table_ndim, const size_t *table_shape, const int64_t *indices, size_t count, size_t axis, size_t elem_bytes, void *out)`

Gather for raw buffers of any element size.

Parameters:

- `table` (`const void *`): Source buffer, row-major, shape table_shape.
- `table_ndim` (`size_t`): Rank of table, [1, 8].
- `table_shape` (`const size_t *`): Shape of table, all dimensions non-zero.
- `indices` (`const int64_t *`): Index array of length count, negative counts from the end of axis (see st_gather).
- `count` (`size_t`): Number of indices, greater than zero.
- `axis` (`size_t`): Gather axis, below table_ndim.
- `elem_bytes` (`size_t`): Element size in bytes (4 for f32/bf16-packed, 8 for int64, 1 for bool).
- `out` (`void *`): Output buffer with outer * count * inner elements.

### `st_permute_copy_into`

`bool st_permute_copy_into(const FloatTensor *src, const size_t *perm, FloatTensor *out)`

Materialize a permuted copy into a prepared output tensor.

Parameters:

- `src` (`const FloatTensor *`): Source tensor, contiguous, f32 or bf16.
- `perm` (`const size_t *`): Permutation where output axis i reads input axis perm[i] (same convention as st_permute and nm_lib's run_transpose).
- `out` (`FloatTensor *`): Output tensor, contiguous, same dtype as src, shape equal to the permuted source shape. Must not alias src (including identity permutations, which would copy onto itself).

### `st_permute_copy_into_bytes`

`bool st_permute_copy_into_bytes(const void *src, size_t ndim, const size_t *shape, const size_t *perm, size_t elem_bytes, void *out)`

Materializing permute for raw buffers of any element size.

Parameters:

- `src` (`const void *`): Source buffer, row-major, shape shape.
- `ndim` (`size_t`): Rank, [1, 8].
- `shape` (`const size_t *`): Source shape, all dimensions non-zero.
- `perm` (`const size_t *`): Permutation array where output axis i reads input axis perm[i].
- `elem_bytes` (`size_t`): Element size in bytes.
- `out` (`void *`): Output buffer with the permuted shape's element count.
