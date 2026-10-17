# `dm.h` – Dense Double Matrix API

Public API for dense double-precision matrices.

## Types

### Typedefs

- `struct DoubleMatrix DoubleMatrix` — Dense double matrix with row-major storage: values[i * cols + j].

## Functions

### `dm_create_empty`

`dm_create_empty(void)`

Create empty matrix metadata (values == NULL).

Parameters:

- `(unnamed)` (`void`)

Returns: Empty matrix metadata (caller must set values), or NULL on allocation failure.

### `dm_create_with_values`

`dm_create_with_values(size_t rows, size_t cols, double *values)`

Create matrix from external values pointer (no copy).

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `values` (`double *`): Pointer to existing row-major storage.

Returns: New matrix wrapping values, or NULL on allocation failure.

### `dm_create`

`dm_create(size_t rows, size_t cols)`

Create uninitialized matrix with shape rows x cols.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.

Returns: New uninitialized matrix, or NULL on allocation failure.

### `dm_clone`

`dm_clone(const DoubleMatrix *m)`

Create deep copy of a matrix.

Parameters:

- `m` (`const DoubleMatrix *`): Source matrix.

Returns: New cloned matrix, or NULL on allocation failure.

### `dm_create_clone`

`dm_create_clone(const DoubleMatrix *m)`

Parameters:

- `m` (`const DoubleMatrix *`)

### `dm_create_identity`

`dm_create_identity(size_t n)`

Create identity matrix with shape n x n.

Parameters:

- `n` (`size_t`): Dimension.

Returns: New identity matrix, or NULL on allocation failure.

### `dm_create_random`

`dm_create_random(size_t rows, size_t cols)`

Create random matrix using global RNG state (not thread-safe).

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.

Returns: New random matrix, or NULL on allocation failure.

### `dm_create_random_seeded`

`dm_create_random_seeded(size_t rows, size_t cols, uint64_t seed)`

Create deterministic random matrix from explicit seed.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `seed` (`uint64_t`): Random seed.

Returns: New random matrix, or NULL on allocation failure.

### `dm_set_random_seed`

`void dm_set_random_seed(uint64_t seed)`

Set global RNG seed used by non-seeded random creators (not thread-safe).

Parameters:

- `seed` (`uint64_t`): Random seed.

### `dm_get_random_seed`

`uint64_t dm_get_random_seed(void)`

Get current global RNG seed.

Parameters:

- `(unnamed)` (`void`)

Returns: Current global RNG seed.

### `dm_from_array_ptrs`

`dm_from_array_ptrs(size_t rows, size_t cols, double **array)`

Create matrix by copying data from row-pointer array.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `array` (`double **`): Array of row pointers.

Returns: New matrix, or NULL on allocation failure.

### `dm_from_array_static`

`dm_from_array_static(size_t rows, size_t cols, double array[rows][cols])`

Create matrix by copying data from static 2D C array.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `array` (`double`): VLA static 2D array.

Returns: New matrix, or NULL on allocation failure.

### `dm_create_from_array`

`dm_create_from_array(size_t rows, size_t cols, double **array)`

Parameters:

- `rows` (`size_t`)
- `cols` (`size_t`)
- `array` (`double **`)

### `dm_create_from_2D_array`

`dm_create_from_2D_array(size_t rows, size_t cols, double array[rows][cols])`

Parameters:

- `rows` (`size_t`)
- `cols` (`size_t`)
- `array` (`double`)

### `dm_get`

`double dm_get(const DoubleMatrix *mat, size_t i, size_t j)`

Read element at (i, j); caller must ensure valid bounds.

Parameters:

- `mat` (`const DoubleMatrix *`): Source matrix.
- `i` (`size_t`): Row index.
- `j` (`size_t`): Column index.

Returns: Element value.

### `dm_set`

`void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value)`

Write element at (i, j); concurrent writes to same matrix are not thread-safe.

Parameters:

- `mat` (`DoubleMatrix *`): Destination matrix.
- `i` (`size_t`): Row index.
- `j` (`size_t`): Column index.
- `value` (`double`): Element value to write.

### `dm_get_row`

`dm_get_row(const DoubleMatrix *mat, size_t i)`

Return row i as new matrix.

Parameters:

- `mat` (`const DoubleMatrix *`): Source matrix.
- `i` (`size_t`): Row index.

Returns: New 1×cols matrix, or NULL on error.

### `dm_get_last_row`

`dm_get_last_row(const DoubleMatrix *mat)`

Return last row as new matrix.

Parameters:

- `mat` (`const DoubleMatrix *`): Source matrix.

Returns: New 1×cols matrix, or NULL on error.

### `dm_get_col`

`dm_get_col(const DoubleMatrix *mat, size_t j)`

Return column j as new matrix.

Parameters:

- `mat` (`const DoubleMatrix *`): Source matrix.
- `j` (`size_t`): Column index.

Returns: New rows×1 matrix, or NULL on error.

### `dm_get_last_col`

`dm_get_last_col(const DoubleMatrix *mat)`

Return last column as new matrix.

Parameters:

- `mat` (`const DoubleMatrix *`): Source matrix.

Returns: New rows×1 matrix, or NULL on error.

### `dm_reshape`

`void dm_reshape(DoubleMatrix *matrix, size_t new_rows, size_t new_cols)`

Reshape matrix metadata; element count must remain compatible.

Parameters:

- `matrix` (`DoubleMatrix *`): Matrix to reshape.
- `new_rows` (`size_t`): New row count.
- `new_cols` (`size_t`): New column count.

### `dm_resize`

`void dm_resize(DoubleMatrix *mat, size_t new_row, size_t new_col)`

Resize matrix storage to new shape.

Parameters:

- `mat` (`DoubleMatrix *`): Matrix to resize.
- `new_row` (`size_t`): New row count.
- `new_col` (`size_t`): New column count.

### `dm_multiply`

`dm_multiply(const DoubleMatrix *mat1, const DoubleMatrix *mat2)`

Multiply two dense double matrices.

Parameters:

- `mat1` (`const DoubleMatrix *`): Left matrix.
- `mat2` (`const DoubleMatrix *`): Right matrix.

Returns: New product matrix, or NULL on error.

### `dm_multiply_by_number`

`dm_multiply_by_number(const DoubleMatrix *mat, const double number)`

Multiply matrix by scalar value.

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.
- `number` (`const double`): Scalar multiplier.

Returns: New scaled matrix, or NULL on error.

### `dm_elementwise_multiply`

`dm_elementwise_multiply(const DoubleMatrix *mat1, const DoubleMatrix *mat2)`

Element-wise product of two matrices.

Parameters:

- `mat1` (`const DoubleMatrix *`): First matrix.
- `mat2` (`const DoubleMatrix *`): Second matrix.

Returns: New element-wise product, or NULL on error.

### `dm_div`

`dm_div(const DoubleMatrix *mat1, const DoubleMatrix *mat2)`

Element-wise division of two matrices.

Parameters:

- `mat1` (`const DoubleMatrix *`): Dividend matrix.
- `mat2` (`const DoubleMatrix *`): Divisor matrix.

Returns: New element-wise quotient, or NULL on error.

### `dm_add`

`dm_add(const DoubleMatrix *mat1, const DoubleMatrix *mat2)`

Add two matrices.

Parameters:

- `mat1` (`const DoubleMatrix *`): First matrix.
- `mat2` (`const DoubleMatrix *`): Second matrix.

Returns: New sum matrix, or NULL on error.

### `dm_diff`

`dm_diff(const DoubleMatrix *mat1, const DoubleMatrix *mat2)`

Subtract two matrices (mat1 - mat2).

Parameters:

- `mat1` (`const DoubleMatrix *`): Minuend matrix.
- `mat2` (`const DoubleMatrix *`): Subtrahend matrix.

Returns: New difference matrix, or NULL on error.

### `dm_inverse`

`dm_inverse(const DoubleMatrix *mat)`

Compute inverse matrix.

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

Returns: New inverse matrix, or NULL on error.

### `dm_transpose`

`dm_transpose(const DoubleMatrix *mat)`

Return transposed copy of a matrix.

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

Returns: New transposed matrix, or NULL on error.

### `dm_inplace_add`

`bool dm_inplace_add(DoubleMatrix *mat1, const DoubleMatrix *mat2)`

In-place matrix addition.

Parameters:

- `mat1` (`DoubleMatrix *`): First matrix (modified).
- `mat2` (`const DoubleMatrix *`): Second matrix.

### `dm_inplace_diff`

`bool dm_inplace_diff(DoubleMatrix *mat1, const DoubleMatrix *mat2)`

In-place matrix subtraction (mat1 -= mat2).

Parameters:

- `mat1` (`DoubleMatrix *`): Minuend matrix (modified).
- `mat2` (`const DoubleMatrix *`): Subtrahend matrix.

### `dm_inplace_transpose`

`bool dm_inplace_transpose(DoubleMatrix *mat)`

In-place transpose (implementation-dependent constraints).

Parameters:

- `mat` (`DoubleMatrix *`): Matrix to transpose (modified in-place).

### `dm_inplace_multiply_by_number`

`bool dm_inplace_multiply_by_number(DoubleMatrix *mat, const double scalar)`

In-place scalar multiplication.

Parameters:

- `mat` (`DoubleMatrix *`): Matrix to scale (modified).
- `scalar` (`const double`): Multiplier value.

### `dm_inplace_gauss_elimination`

`bool dm_inplace_gauss_elimination(DoubleMatrix *mat)`

In-place Gaussian elimination transform.

Parameters:

- `mat` (`DoubleMatrix *`): Matrix to transform (modified).

### `dm_inplace_elementwise_multiply`

`bool dm_inplace_elementwise_multiply(DoubleMatrix *mat1, const DoubleMatrix *mat2)`

In-place element-wise multiplication.

Parameters:

- `mat1` (`DoubleMatrix *`): First matrix (modified).
- `mat2` (`const DoubleMatrix *`): Second matrix.

### `dm_inplace_div`

`bool dm_inplace_div(DoubleMatrix *mat1, const DoubleMatrix *mat2)`

In-place element-wise division.

Parameters:

- `mat1` (`DoubleMatrix *`): Dividend matrix (modified).
- `mat2` (`const DoubleMatrix *`): Divisor matrix.

### `dm_determinant`

`double dm_determinant(const DoubleMatrix *mat)`

Determinant of a square matrix.

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

Returns: Determinant value.

### `dm_trace`

`double dm_trace(const DoubleMatrix *mat)`

Trace of a square matrix.

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

Returns: Trace value (sum of diagonal elements).

### `dm_rank`

`size_t dm_rank(const DoubleMatrix *mat)`

Matrix rank.

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

Returns: Numerical rank.

### `dm_norm`

`double dm_norm(const DoubleMatrix *mat)`

Matrix norm (implementation-defined norm type).

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

Returns: Norm value.

### `dm_density`

`double dm_density(const DoubleMatrix *mat)`

Matrix density in range [0,1].

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

Returns: Density value (approx. ratio of non-zero elements).

### `dm_is_empty`

`bool dm_is_empty(const DoubleMatrix *mat)`

Check whether matrix is empty/uninitialized.

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

### `dm_is_square`

`bool dm_is_square(const DoubleMatrix *mat)`

Check whether matrix is square.

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

### `dm_is_vector`

`bool dm_is_vector(const DoubleMatrix *mat)`

Check whether matrix represents a vector (one dimension equals 1).

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

### `dm_is_equal_size`

`bool dm_is_equal_size(const DoubleMatrix *mat1, const DoubleMatrix *mat2)`

Check whether two matrices have equal shape.

Parameters:

- `mat1` (`const DoubleMatrix *`): First matrix.
- `mat2` (`const DoubleMatrix *`): Second matrix.

### `dm_is_equal`

`bool dm_is_equal(const DoubleMatrix *mat1, const DoubleMatrix *mat2)`

Check whether two matrices are element-wise equal.

Parameters:

- `mat1` (`const DoubleMatrix *`): First matrix.
- `mat2` (`const DoubleMatrix *`): Second matrix.

### `dm_to_column_major`

`double * dm_to_column_major(const DoubleMatrix *mat)`

Export matrix as newly allocated column-major array (caller must free).

Parameters:

- `mat` (`const DoubleMatrix *`): Input matrix.

Returns: Newly allocated array, or NULL on error.

### `dm_print`

`void dm_print(const DoubleMatrix *matrix)`

Print matrix to stdout (debug helper).

Parameters:

- `matrix` (`const DoubleMatrix *`): Matrix to print.

### `dm_active_library`

`const char * dm_active_library(void)`

Return active compute backend name.

Parameters:

- `(unnamed)` (`void`)

Returns: Backend name string (statically allocated).

### `dm_destroy`

`void dm_destroy(DoubleMatrix *mat)`

Destroy matrix (NULL-safe).

Parameters:

- `mat` (`DoubleMatrix *`): Matrix to destroy.
