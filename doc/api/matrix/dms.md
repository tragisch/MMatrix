# `dms.h` – Sparse Double Matrix API

Public API for sparse double matrices in COO format.

## Types

### Typedefs

- `struct cs_di_sparse cs` — Forward declaration for SuiteSparse type when cs.h is unavailable.
- `struct DoubleSparseMatrix DoubleSparseMatrix` — Sparse COO matrix stored as (row, col, value) triples.

## Functions

### `dms_create_empty`

`dms_create_empty(void)`

Create empty sparse matrix metadata (nnz = 0, arrays NULL).

Parameters:

- `(unnamed)` (`void`)

Returns: Empty sparse matrix, or NULL on allocation failure.

### `dms_create_with_values`

`dms_create_with_values(size_t rows, size_t cols, size_t nnz, size_t *row_indices, size_t *col_indices, double *values)`

Create sparse matrix by copying external COO arrays.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `nnz` (`size_t`): Number of non-zero entries.
- `row_indices` (`size_t *`): Array of row indices (length nnz).
- `col_indices` (`size_t *`): Array of column indices (length nnz).
- `values` (`double *`): Array of values (length nnz).

Returns: Sparse matrix with copied COO data, or NULL on allocation failure.

### `dms_create`

`dms_create(size_t rows, size_t cols, size_t capacity)`

Create empty COO matrix with pre-allocated triple capacity.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `capacity` (`size_t`): Pre-allocated capacity for nnz triples.

Returns: Empty sparse matrix with allocated arrays, or NULL on allocation failure.

### `dms_clone`

`dms_clone(const DoubleSparseMatrix *m)`

Create deep copy of sparse matrix.

Parameters:

- `m` (`const DoubleSparseMatrix *`): Source sparse matrix.

Returns: New cloned sparse matrix, or NULL on allocation failure.

### `dms_create_clone`

`dms_create_clone(const DoubleSparseMatrix *m)`

DeprecatedUse dms_clone instead. m Source sparse matrix. New cloned sparse matrix, or NULL on allocation failure.

Parameters:

- `m` (`const DoubleSparseMatrix *`): Source sparse matrix.

Returns: New cloned sparse matrix, or NULL on allocation failure.

### `dms_create_identity`

`dms_create_identity(size_t n)`

Create sparse identity matrix with shape n x n.

Parameters:

- `n` (`size_t`): Dimension (rows and columns).

Returns: Identity sparse matrix, or NULL on allocation failure.

### `dms_create_random`

`dms_create_random(size_t rows, size_t cols, double density)`

Create sparse random matrix using shared global RNG state.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `density` (`double`): Target sparsity as fraction in [0,1].

Returns: Random sparse matrix, or NULL on allocation failure.

### `dms_create_random_seeded`

`dms_create_random_seeded(size_t rows, size_t cols, double density, uint64_t seed)`

Create deterministic sparse random matrix from explicit seed.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `density` (`double`): Target sparsity as fraction in [0,1].
- `seed` (`uint64_t`): Random seed for reproducibility.

Returns: Deterministic random sparse matrix, or NULL on allocation failure.

### `dms_set_random_seed`

`void dms_set_random_seed(uint64_t seed)`

Set shared global RNG seed used by non-seeded random creators.

Parameters:

- `seed` (`uint64_t`): Random seed value.

### `dms_get_random_seed`

`uint64_t dms_get_random_seed(void)`

Get current global RNG seed.

Parameters:

- `(unnamed)` (`void`)

Returns: Current global RNG seed value.

### `dms_to_cs`

`dms_to_cs(const DoubleSparseMatrix *coo)`

Convert COO sparse matrix to SuiteSparse cs matrix.

Parameters:

- `coo` (`const DoubleSparseMatrix *`): Source COO matrix.

Returns: New SuiteSparse cs matrix, or NULL on allocation failure.

### `dms_from_cs`

`dms_from_cs(const cs *A)`

Convert SuiteSparse cs matrix to COO sparse matrix.

Parameters:

- `A` (`const cs *`): Source SuiteSparse cs matrix.

Returns: New COO sparse matrix, or NULL on allocation failure.

### `cs_to_dms`

`cs_to_dms(const cs *A)`

DeprecatedUse dms_from_cs instead. A Source SuiteSparse cs matrix. New COO sparse matrix, or NULL on allocation failure.

Parameters:

- `A` (`const cs *`): Source SuiteSparse cs matrix.

Returns: New COO sparse matrix, or NULL on allocation failure.

### `dms_to_array`

`double * dms_to_array(const DoubleSparseMatrix *mat)`

Export sparse matrix as newly allocated dense row-major array.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix.

Returns: Newly allocated dense array (size rows x cols), or NULL on allocation failure. Caller must free.

### `dms_create_from_array`

`dms_create_from_array(size_t rows, size_t cols, double *array)`

Create sparse matrix from dense row-major array.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `array` (`double *`): Dense row-major array of size rows x cols.

Returns: Sparse matrix with zero entries trimmed, or NULL on allocation failure.

### `dms_from_array_static`

`dms_from_array_static(size_t rows, size_t cols, double array[rows][cols])`

Create sparse matrix from static dense 2D C array.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `array` (`double`): VLA static 2D array of shape rows x cols.

Returns: Sparse matrix with zero entries trimmed, or NULL on allocation failure.

### `dms_create_from_2D_array`

`dms_create_from_2D_array(size_t rows, size_t cols, double array[rows][cols])`

DeprecatedUse dms_from_array_static instead. rows Number of rows. cols Number of columns. array VLA static 2D array. Sparse matrix, or NULL on allocation failure.

Parameters:

- `rows` (`size_t`): Number of rows.
- `cols` (`size_t`): Number of columns.
- `array` (`double`): VLA static 2D array.

Returns: Sparse matrix, or NULL on allocation failure.

### `dms_set`

`bool dms_set(DoubleSparseMatrix *mat, size_t i, size_t j, double value)`

Insert or update COO triple (i, j, value).

Parameters:

- `mat` (`DoubleSparseMatrix *`): Sparse matrix to update.
- `i` (`size_t`): Row index (0-based, must be < rows).
- `j` (`size_t`): Column index (0-based, must be < cols).
- `value` (`double`): New value (replaces or adds to existing).

Returns: true on success, false on allocation/bound error.

### `dms_get`

`double dms_get(const DoubleSparseMatrix *mat, size_t i, size_t j)`

Read value at first matching (i, j) position.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix.
- `i` (`size_t`): Row index (0-based, must be < rows).
- `j` (`size_t`): Column index (0-based, must be < cols).

Returns: Value at (i, j), or 0.0 if no entry exists.

### `dms_get_row`

`dms_get_row(const DoubleSparseMatrix *mat, size_t i)`

Return row i as new sparse matrix.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix.
- `i` (`size_t`): Row index (0-based, must be < rows).

Returns: New sparse 1 x cols matrix, or NULL on allocation failure.

### `dms_get_last_row`

`dms_get_last_row(const DoubleSparseMatrix *mat)`

Return last row as new sparse matrix.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix.

Returns: New sparse 1 x cols matrix, or NULL on allocation failure.

### `dms_get_col`

`dms_get_col(const DoubleSparseMatrix *mat, size_t j)`

Return column j as new sparse matrix.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix.
- `j` (`size_t`): Column index (0-based, must be < cols).

Returns: New sparse rows x 1 matrix, or NULL on allocation failure.

### `dms_get_last_col`

`dms_get_last_col(const DoubleSparseMatrix *mat)`

Return last column as new sparse matrix.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix.

Returns: New sparse rows x 1 matrix, or NULL on allocation failure.

### `dms_multiply`

`dms_multiply(const DoubleSparseMatrix *mat1, const DoubleSparseMatrix *mat2)`

Multiply two sparse matrices.

Parameters:

- `mat1` (`const DoubleSparseMatrix *`): Left sparse matrix (shape: m x k).
- `mat2` (`const DoubleSparseMatrix *`): Right sparse matrix (shape: k x n).

Returns: Result sparse matrix of shape m x n, or NULL on allocation/shape mismatch.

### `dms_multiply_by_number`

`dms_multiply_by_number(const DoubleSparseMatrix *mat, const double number)`

Multiply sparse matrix by scalar value.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix.
- `number` (`const double`): Scalar multiplier.

Returns: Result sparse matrix, or NULL on allocation failure.

### `dms_transpose`

`dms_transpose(const DoubleSparseMatrix *mat)`

Return transposed sparse matrix.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix (shape: rows x cols).

Returns: Transposed sparse matrix (shape: cols x rows), or NULL on allocation failure.

### `dms_spmv`

`bool dms_spmv(const DoubleSparseMatrix *mat, const double *x, double *y)`

Sparse matrix-vector product y = A * x.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix (shape: rows x cols).
- `x` (`const double *`): Input vector (length cols).
- `y` (`double *`): Output vector (length rows); pre-allocated by caller.

Returns: true on success, false on dimension/allocation error.

### `dms_density`

`double dms_density(const DoubleSparseMatrix *mat)`

Matrix density in range [0,1].

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Source sparse matrix.

Returns: Ratio of non-zero entries to total entries (nnz / (rows * cols)).

### `dms_print`

`void dms_print(const DoubleSparseMatrix *mat)`

Print sparse matrix to stdout (debug helper).

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Matrix to print.

### `dms_realloc`

`bool dms_realloc(DoubleSparseMatrix *mat, size_t new_capacity)`

Reallocate COO arrays to new_capacity (new_capacity >= nnz).

Parameters:

- `mat` (`DoubleSparseMatrix *`): Sparse matrix to reallocate.
- `new_capacity` (`size_t`): New capacity for triples.

Returns: true on success, false on allocation failure.

### `dms_destroy`

`void dms_destroy(DoubleSparseMatrix *mat)`

Destroy sparse matrix (NULL-safe).

Parameters:

- `mat` (`DoubleSparseMatrix *`): Sparse matrix pointer (NULL-safe; no-op if NULL).
