# `m_convert.h` – Dense/Sparse Matrix Conversion API

Public conversion API between dense and sparse matrix types.

## Functions

### `dms_to_dm`

`dms_to_dm(const DoubleSparseMatrix *src)`

Convert sparse COO double matrix to dense double matrix.

Parameters:

- `src` (`const DoubleSparseMatrix *`): Source sparse matrix.

Returns: Dense matrix with all COO entries expanded, or NULL on allocation failure.

### `dm_to_dms`

`dm_to_dms(const DoubleMatrix *src)`

Convert dense double matrix to sparse COO double matrix.

Parameters:

- `src` (`const DoubleMatrix *`): Source dense matrix.

Returns: Sparse matrix with zero entries removed, or NULL on allocation failure.

### `sm_to_dm`

`sm_to_dm(const FloatMatrix *sm)`

Convert dense float matrix to dense double matrix.

Parameters:

- `sm` (`const FloatMatrix *`): Source float matrix.

Returns: Dense double matrix with precision-expanded values, or NULL on allocation failure.

### `dm_to_sm`

`dm_to_sm(const DoubleMatrix *src)`

Convert dense double matrix to dense float matrix.

Parameters:

- `src` (`const DoubleMatrix *`): Source double matrix.

Returns: Dense float matrix with precision-reduced values, or NULL on allocation failure.
