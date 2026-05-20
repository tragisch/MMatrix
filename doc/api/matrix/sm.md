# `sm.h` – Dense Float Matrix API

Public API for dense single-precision matrices and backend-dispatched kernels.

## Types

### `SmBackend`

Available compute backends (runtime selection).

Values:

- `SM_BACKEND_DEFAULT`
- `SM_BACKEND_ACCELERATE`
- `SM_BACKEND_OPENBLAS`
- `SM_BACKEND_OPENMP`

### `SmTranspose`

Transpose mode for BLAS-style operations.

Values:

- `SM_NO_TRANSPOSE`
- `SM_TRANSPOSE`

### Typedefs

- `enum SmBackend` — Available compute backends (runtime selection).
- `struct FloatMatrix FloatMatrix` — Dense float matrix with row-major storage: values[i * cols + j].
- `enum SmTranspose` — Transpose mode for BLAS-style operations.

## Functions

### `sm_set_backend`

`bool sm_set_backend(SmBackend backend)`

Set active compute backend for dispatched operations.

Parameters:

- `backend` (`SmBackend`)

### `sm_get_backend`

`sm_get_backend(void)`

Get currently active compute backend.

Parameters:

- `(unnamed)` (`void`)

Returns: Active backend enum value.

### `sm_create_empty`

`sm_create_empty(void)`

Create empty matrix metadata (values == NULL).

Parameters:

- `(unnamed)` (`void`)

Returns: Empty matrix with uninitialized rows/cols, or NULL on allocation failure.

### `sm_create_zeros`

`sm_create_zeros(size_t rows, size_t cols)`

Create zero-initialized matrix with shape rows x cols.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.

Returns: New zero matrix, or NULL on allocation failure.

### `sm_create`

`sm_create(size_t rows, size_t cols)`

Create uninitialized matrix with shape rows x cols.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.

Returns: New uninitialized matrix, or NULL on allocation failure.

### `sm_create_with_values`

`sm_create_with_values(size_t rows, size_t cols, float *values)`

Create matrix by copying provided values array.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `values` (`float *`): Pointer to row-major float array of size rows * cols.

Returns: New matrix with copied values, or NULL on allocation failure.

### `sm_clone`

`sm_clone(const FloatMatrix *m)`

Create deep copy of a matrix.

Parameters:

- `m` (`const FloatMatrix *`): Source matrix pointer.

Returns: New cloned matrix, or NULL on allocation failure.

### `sm_create_identity`

`sm_create_identity(size_t n)`

Create identity matrix with shape n x n.

Parameters:

- `n` (`size_t`): Dimension (rows and columns).

Returns: Identity matrix, or NULL on allocation failure.

### `sm_create_random`

`sm_create_random(size_t rows, size_t cols)`

Create random matrix using global RNG state (not thread-safe).

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.

Returns: Random matrix, or NULL on allocation failure.

### `sm_create_random_seeded`

`sm_create_random_seeded(size_t rows, size_t cols, uint64_t seed)`

Create deterministic random matrix from explicit seed.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `seed` (`uint64_t`): Random seed for reproducibility.

Returns: Random matrix, or NULL on allocation failure.

### `sm_set_random_seed`

`void sm_set_random_seed(uint64_t seed)`

Set global RNG seed used by non-seeded creators (not thread-safe).

Parameters:

- `seed` (`uint64_t`): Random seed value.

### `sm_get_random_seed`

`uint64_t sm_get_random_seed(void)`

Get current global RNG seed.

Parameters:

- `(unnamed)` (`void`)

Returns: Current global RNG seed value.

### `sm_create_random_he`

`sm_create_random_he(size_t rows, size_t cols, size_t fan_in)`

Create He-initialized weight matrix (ReLU-like networks).

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `fan_in` (`size_t`): Fan-in (typically input dimension).

Returns: He-initialized weight matrix, or NULL on allocation failure.

### `sm_create_random_xavier`

`sm_create_random_xavier(size_t rows, size_t cols, size_t fan_in, size_t fan_out)`

Create Xavier/Glorot-initialized weight matrix.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `fan_in` (`size_t`): Fan-in (input dimension).
- `fan_out` (`size_t`): Fan-out (output dimension).

Returns: Xavier-initialized weight matrix, or NULL on allocation failure.

### `sm_from_array_ptrs`

`sm_from_array_ptrs(size_t rows, size_t cols, float **array)`

Create matrix by copying data from row-pointer array.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `array` (`float **`): Array of row pointers.

Returns: New matrix with copied data, or NULL on allocation failure.

### `sm_from_array_static`

`sm_from_array_static(size_t rows, size_t cols, float array[rows][cols])`

Create matrix by copying data from static 2D C array.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `array` (`float`): VLA static 2D array of shape rows x cols.

Returns: New matrix with copied data, or NULL on allocation failure.

### `sm_to_array`

`float * sm_to_array(FloatMatrix *matrix)`

Export matrix to newly allocated row-major array (caller must free).

Parameters:

- `matrix` (`FloatMatrix *`): Source matrix.

Returns: Newly allocated float array in row-major order, or NULL on allocation failure.

### `sm_create_array_from_matrix`

`float * sm_create_array_from_matrix(FloatMatrix *matrix)`

DeprecatedUse sm_to_array instead. matrix Source matrix. Newly allocated float array, or NULL on allocation failure.

Parameters:

- `matrix` (`FloatMatrix *`): Source matrix.

Returns: Newly allocated float array, or NULL on allocation failure.

### `sm_get`

`float sm_get(const FloatMatrix *mat, size_t i, size_t j)`

Read element at (i, j); caller must ensure valid bounds.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.
- `i` (`size_t`): Row index (0-based, must be < rows).
- `j` (`size_t`): Column index (0-based, must be < cols).

Returns: Element value at position (i, j).

### `sm_set`

`void sm_set(FloatMatrix *mat, size_t i, size_t j, float value)`

Write element at (i, j); concurrent writes are not thread-safe.

Parameters:

- `mat` (`FloatMatrix *`): Destination matrix.
- `i` (`size_t`): Row index (0-based, must be < rows).
- `j` (`size_t`): Column index (0-based, must be < cols).
- `value` (`float`): New element value.

### `sm_get_row`

`sm_get_row(const FloatMatrix *mat, size_t i)`

Return row i as new matrix.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.
- `i` (`size_t`): Row index (0-based, must be < rows).

Returns: New 1 x cols matrix, or NULL on allocation failure.

### `sm_get_last_row`

`sm_get_last_row(const FloatMatrix *mat)`

Return last row as new matrix.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.

Returns: New 1 x cols matrix, or NULL on allocation failure.

### `sm_get_col`

`sm_get_col(const FloatMatrix *mat, size_t j)`

Return column j as new matrix.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.
- `j` (`size_t`): Column index (0-based, must be < cols).

Returns: New rows x 1 matrix, or NULL on allocation failure.

### `sm_get_last_col`

`sm_get_last_col(const FloatMatrix *mat)`

Return last column as new matrix.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.

Returns: New rows x 1 matrix, or NULL on allocation failure.

### `sm_slice_rows`

`sm_slice_rows(const FloatMatrix *mat, size_t start, size_t end)`

Return row slice [start, end) as new matrix.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.
- `start` (`size_t`): Starting row index (inclusive, 0-based).
- `end` (`size_t`): Ending row index (exclusive, 0-based).

Returns: New (end-start) x cols matrix, or NULL on allocation/range error.

### `sm_reshape`

`void sm_reshape(FloatMatrix *matrix, size_t new_rows, size_t new_cols)`

Reshape matrix metadata; element count must remain compatible.

Parameters:

- `matrix` (`FloatMatrix *`): Matrix to reshape (metadata only, not realloc).
- `new_rows` (`size_t`): New number of rows.
- `new_cols` (`size_t`): New number of columns.

### `sm_resize`

`void sm_resize(FloatMatrix *mat, size_t new_row, size_t new_col)`

Resize matrix storage to new shape.

Parameters:

- `mat` (`FloatMatrix *`): Matrix to resize (may reallocate).
- `new_row` (`size_t`): New number of rows.
- `new_col` (`size_t`): New number of columns.

### `sm_transpose`

`sm_transpose(const FloatMatrix *mat)`

Return transposed copy of matrix.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix (shape: rows x cols).

Returns: Transposed matrix (shape: cols x rows), or NULL on allocation failure.

### `sm_add`

`sm_add(const FloatMatrix *mat1, const FloatMatrix *mat2)`

Add two matrices element-wise.

Parameters:

- `mat1` (`const FloatMatrix *`): First matrix.
- `mat2` (`const FloatMatrix *`): Second matrix (must have same shape as mat1).

Returns: Result matrix, or NULL on allocation/shape mismatch.

### `sm_diff`

`sm_diff(const FloatMatrix *mat1, const FloatMatrix *mat2)`

Subtract two matrices element-wise (mat1 - mat2).

Parameters:

- `mat1` (`const FloatMatrix *`): First matrix.
- `mat2` (`const FloatMatrix *`): Second matrix (must have same shape as mat1).

Returns: Result matrix, or NULL on allocation/shape mismatch.

### `sm_multiply`

`sm_multiply(const FloatMatrix *mat1, const FloatMatrix *mat2)`

Multiply two matrices (standard matrix multiplication).

Parameters:

- `mat1` (`const FloatMatrix *`): Left matrix (shape: m x k).
- `mat2` (`const FloatMatrix *`): Right matrix (shape: k x n).

Returns: Result matrix of shape m x n, or NULL on allocation/shape mismatch.

### `sm_elementwise_multiply`

`sm_elementwise_multiply(const FloatMatrix *mat1, const FloatMatrix *mat2)`

Element-wise product of two matrices.

Parameters:

- `mat1` (`const FloatMatrix *`): First matrix.
- `mat2` (`const FloatMatrix *`): Second matrix (must have same shape as mat1).

Returns: Result matrix, or NULL on allocation/shape mismatch.

### `sm_multiply_by_number`

`sm_multiply_by_number(const FloatMatrix *mat, const float number)`

Multiply matrix by scalar value.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.
- `number` (`const float`): Scalar multiplier.

Returns: Result matrix, or NULL on allocation failure.

### `sm_inverse`

`sm_inverse(const FloatMatrix *mat)`

Compute inverse matrix.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix (must be square and invertible).

Returns: Inverse matrix, or NULL on allocation/singularity error.

### `sm_div`

`sm_div(const FloatMatrix *mat1, const FloatMatrix *mat2)`

Element-wise division of two matrices.

Parameters:

- `mat1` (`const FloatMatrix *`): Numerator matrix.
- `mat2` (`const FloatMatrix *`): Denominator matrix (must have same shape as mat1).

Returns: Result matrix, or NULL on allocation/shape/zero-division error.

### `sm_solve_system`

`sm_solve_system(const FloatMatrix *A, const FloatMatrix *b)`

Solve linear system A * x = b.

Parameters:

- `A` (`const FloatMatrix *`): Coefficient matrix (must be square and invertible).
- `b` (`const FloatMatrix *`): Right-hand-side vector (shape: rows x 1).

Returns: Solution vector x, or NULL on allocation/singularity error.

### `sm_gemm`

`bool sm_gemm(FloatMatrix *C, float alpha, const FloatMatrix *A, SmTranspose trans_a, const FloatMatrix *B, SmTranspose trans_b, float beta)`

BLAS-style GEMM kernel.

Parameters:

- `C` (`FloatMatrix *`): Output matrix (shape must be compatible with A and B).
- `alpha` (`float`): Scalar multiplier for A*B term.
- `A` (`const FloatMatrix *`): Left input matrix.
- `trans_a` (`SmTranspose`): Transpose mode for A (SM_NO_TRANSPOSE or SM_TRANSPOSE).
- `B` (`const FloatMatrix *`): Right input matrix.
- `trans_b` (`SmTranspose`): Transpose mode for B.
- `beta` (`float`): Scalar multiplier for existing C values.

Returns: true on success, false on shape mismatch or allocation failure.

### `sm_gemm_bias_relu`

`bool sm_gemm_bias_relu(FloatMatrix *C, const FloatMatrix *A, SmTranspose trans_a, const FloatMatrix *B, SmTranspose trans_b, const FloatMatrix *bias)`

Fused kernel: GEMM + optional bias + ReLU activation.

Parameters:

- `C` (`FloatMatrix *`): Output matrix to accumulate result into.
- `A` (`const FloatMatrix *`): Left input matrix.
- `trans_a` (`SmTranspose`): Transpose mode for A.
- `B` (`const FloatMatrix *`): Right input matrix.
- `trans_b` (`SmTranspose`): Transpose mode for B.
- `bias` (`const FloatMatrix *`): Bias vector (shape 1 x cols or rows x cols, or NULL to skip).

Returns: true on success, false on shape/allocation error.

### `sm_inplace_add`

`bool sm_inplace_add(FloatMatrix *mat1, const FloatMatrix *mat2)`

In-place matrix addition.

Parameters:

- `mat1` (`FloatMatrix *`): Matrix to accumulate into.
- `mat2` (`const FloatMatrix *`): Matrix to add (must have same shape as mat1).

Returns: true on success, false on shape/allocation error.

### `sm_inplace_diff`

`bool sm_inplace_diff(FloatMatrix *mat1, const FloatMatrix *mat2)`

In-place matrix subtraction (mat1 -= mat2).

Parameters:

- `mat1` (`FloatMatrix *`): Matrix to subtract from.
- `mat2` (`const FloatMatrix *`): Matrix to subtract (must have same shape as mat1).

Returns: true on success, false on shape/allocation error.

### `sm_inplace_square_transpose`

`bool sm_inplace_square_transpose(FloatMatrix *mat)`

In-place transpose for square matrices.

Parameters:

- `mat` (`FloatMatrix *`): Square matrix to transpose.

Returns: true on success, false if matrix is not square.

### `sm_inplace_multiply_by_number`

`bool sm_inplace_multiply_by_number(FloatMatrix *mat, const float scalar)`

In-place scalar multiplication.

Parameters:

- `mat` (`FloatMatrix *`): Matrix to scale.
- `scalar` (`const float`): Multiplicative factor.

Returns: true on success, false on error.

### `sm_inplace_elementwise_multiply`

`bool sm_inplace_elementwise_multiply(FloatMatrix *mat1, const FloatMatrix *mat2)`

In-place element-wise multiplication.

Parameters:

- `mat1` (`FloatMatrix *`): Matrix to scale element-wise.
- `mat2` (`const FloatMatrix *`): Element-wise scale matrix (must have same shape as mat1).

Returns: true on success, false on shape/allocation error.

### `sm_inplace_div`

`bool sm_inplace_div(FloatMatrix *mat1, const FloatMatrix *mat2)`

In-place element-wise division.

Parameters:

- `mat1` (`FloatMatrix *`): Numerator matrix to divide.
- `mat2` (`const FloatMatrix *`): Denominator matrix (must have same shape as mat1).

Returns: true on success, false on shape/zero-division error.

### `sm_inplace_normalize_rows`

`bool sm_inplace_normalize_rows(FloatMatrix *mat)`

In-place row-wise normalization (L2 norm).

Parameters:

- `mat` (`FloatMatrix *`): Matrix to normalize (each row becomes unit length).

Returns: true on success, false on error.

### `sm_inplace_normalize_cols`

`bool sm_inplace_normalize_cols(FloatMatrix *mat)`

In-place column-wise normalization (L2 norm).

Parameters:

- `mat` (`FloatMatrix *`): Matrix to normalize (each column becomes unit length).

Returns: true on success, false on error.

### `sm_determinant`

`float sm_determinant(const FloatMatrix *mat)`

Determinant of a square matrix.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix (must be square).

Returns: Determinant value, or NaN on error/singular matrix.

### `sm_trace`

`float sm_trace(const FloatMatrix *mat)`

Trace of a square matrix (sum of diagonal elements).

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix (must be square).

Returns: Trace value.

### `sm_norm`

`float sm_norm(const FloatMatrix *mat)`

Matrix norm (Frobenius norm).

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.

Returns: Frobenius norm (square root of sum of squared elements).

### `sm_rank`

`size_t sm_rank(const FloatMatrix *mat)`

Matrix rank.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.

Returns: Approximate rank (number of non-negligible singular values).

### `sm_density`

`float sm_density(const FloatMatrix *mat)`

Matrix density in range [0,1].

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.

Returns: Ratio of non-zero elements to total elements.

### `sm_is_empty`

`bool sm_is_empty(const FloatMatrix *mat)`

Check whether matrix is empty/uninitialized.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.

### `sm_is_square`

`bool sm_is_square(const FloatMatrix *mat)`

Check whether matrix is square.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.

### `sm_is_vector`

`bool sm_is_vector(const FloatMatrix *mat)`

Check whether matrix represents a vector.

Parameters:

- `mat` (`const FloatMatrix *`): Source matrix.

### `sm_is_equal_size`

`bool sm_is_equal_size(const FloatMatrix *mat1, const FloatMatrix *mat2)`

Check whether two matrices have equal shape.

Parameters:

- `mat1` (`const FloatMatrix *`): First matrix.
- `mat2` (`const FloatMatrix *`): Second matrix.

### `sm_is_equal`

`bool sm_is_equal(const FloatMatrix *mat1, const FloatMatrix *mat2)`

Check whether two matrices are element-wise equal.

Parameters:

- `mat1` (`const FloatMatrix *`): First matrix.
- `mat2` (`const FloatMatrix *`): Second matrix (must have same shape as mat1).

### `sm_lu_decompose`

`bool sm_lu_decompose(FloatMatrix *mat, size_t *pivot_order)`

Perform LU decomposition (in-place); writes pivot order.

Parameters:

- `mat` (`FloatMatrix *`): Matrix to decompose (overwritten with LU result).
- `pivot_order` (`size_t *`): Array of length rows to store pivot indices.

Returns: true on success, false on allocation/singular matrix error.

### `sm_print`

`void sm_print(const FloatMatrix *matrix)`

Print matrix to stdout (debug helper).

Parameters:

- `matrix` (`const FloatMatrix *`): Matrix to print.

### `sm_active_library`

`const char * sm_active_library(void)`

Return active compute backend name.

Parameters:

- `(unnamed)` (`void`)

Returns: String name of active backend (e.g., "Accelerate", "OpenBLAS").

### `sm_destroy`

`void sm_destroy(FloatMatrix *mat)`

Destroy matrix and release all allocated memory.

Parameters:

- `mat` (`FloatMatrix *`): Matrix pointer (NULL-safe; no-op if NULL).
