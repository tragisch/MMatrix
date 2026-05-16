# `m_io.h` – Matrix I/O API

Public I/O and plotting API for matrix types (MAT + MatrixMarket).

## Types

### `MIOFormat`

C API enums keep default underlying type for ABI stability.

Values:

- `MIO_FMT_MAT5`
- `MIO_FMT_MAT73`

### `MIOCompression`

MATLAB file compression mode.

Values:

- `MIO_COMPRESS_NONE`
- `MIO_COMPRESS_ZLIB`

### `MioStatus`

I/O-specific status codes.

Values:

- `MIO_STATUS_OK`
- `MIO_STATUS_INVALID_ARGUMENT`
- `MIO_STATUS_IO_ERROR`
- `MIO_STATUS_ALLOC_FAILED`
- `MIO_STATUS_FORMAT_ERROR`
- `MIO_STATUS_UNSUPPORTED_TYPE`
- `MIO_STATUS_INTERNAL_ERROR`

### Typedefs

- `enum MioStatus` — I/O-specific status codes.

## Functions

### `dm_cplot`

`void dm_cplot(DoubleMatrix *mat)`

Plot dense double matrix in terminal.

Parameters:

- `mat` (`DoubleMatrix *`): Matrix to display.

### `sm_cplot`

`void sm_cplot(FloatMatrix *mat)`

Plot dense float matrix in terminal.

Parameters:

- `mat` (`FloatMatrix *`): Matrix to display.

### `dms_cplot`

`void dms_cplot(DoubleSparseMatrix *mat, double strength)`

Plot sparse double matrix in terminal.

Parameters:

- `mat` (`DoubleSparseMatrix *`): Sparse matrix to display.
- `strength` (`double`): Color intensity scale factor (typically 1.0).

### `mio_status_to_string`

`const char * mio_status_to_string(MioStatus status)`

Return canonical string for an I/O status code.

Parameters:

- `status` (`MioStatus`): Status code to convert.

Returns: Human-readable status string (static, caller must not free).

### `mio_set_format`

`void mio_set_format(MIOFormat fmt)`

Set global MATLAB file format for subsequent writes (not thread-safe).

Parameters:

- `fmt` (`MIOFormat`): MATLAB file format (MIO_FMT_MAT5 or MIO_FMT_MAT73).

### `mio_get_format`

`mio_get_format(void)`

Get current global MATLAB file format.

Parameters:

- `(unnamed)` (`void`)

Returns: Current MATLAB file format setting.

### `mio_set_compression`

`void mio_set_compression(MIOCompression comp)`

Set global MATLAB compression mode for writes (not thread-safe).

Parameters:

- `comp` (`MIOCompression`): Compression mode (MIO_COMPRESS_NONE or MIO_COMPRESS_ZLIB).

### `mio_get_compression`

`mio_get_compression(void)`

Get current global MATLAB compression mode.

Parameters:

- `(unnamed)` (`void`)

Returns: Current compression mode setting.

### `dm_write_mat_file`

`int dm_write_mat_file(const DoubleMatrix *matrix, const char *filename)`

Legacy MAT writer for DoubleMatrix (0 on success).

Parameters:

- `matrix` (`const DoubleMatrix *`): Matrix to write.
- `filename` (`const char *`): Output file path.

Returns: 0 on success, non-zero error code on failure.

### `dm_write_MAT_file`

`int dm_write_MAT_file(const DoubleMatrix *matrix, const char *filename)`

DeprecatedUse dm_write_mat_file instead. matrix Matrix to write. filename Output file path. 0 on success, non-zero error code on failure.

Parameters:

- `matrix` (`const DoubleMatrix *`): Matrix to write.
- `filename` (`const char *`): Output file path.

Returns: 0 on success, non-zero error code on failure.

### `sm_write_mat_file`

`int sm_write_mat_file(const FloatMatrix *matrix, const char *filename)`

Legacy MAT writer for FloatMatrix (0 on success).

Parameters:

- `matrix` (`const FloatMatrix *`): Matrix to write.
- `filename` (`const char *`): Output file path.

Returns: 0 on success, non-zero error code on failure.

### `sm_write_MAT_file`

`int sm_write_MAT_file(const FloatMatrix *matrix, const char *filename)`

DeprecatedUse sm_write_mat_file instead. matrix Matrix to write. filename Output file path. 0 on success, non-zero error code on failure.

Parameters:

- `matrix` (`const FloatMatrix *`): Matrix to write.
- `filename` (`const char *`): Output file path.

Returns: 0 on success, non-zero error code on failure.

### `dm_read_mat_file`

`dm_read_mat_file(const char *filename)`

Legacy MAT reader for DoubleMatrix (returns NULL on error).

Parameters:

- `filename` (`const char *`): Input MAT file path.

Returns: Pointer to read matrix, or NULL on error. Caller must free with dm_destroy.

### `dm_read_MAT_file`

`dm_read_MAT_file(const char *filename)`

DeprecatedUse dm_read_mat_file instead. filename Input MAT file path. Pointer to read matrix, or NULL on error.

Parameters:

- `filename` (`const char *`): Input MAT file path.

Returns: Pointer to read matrix, or NULL on error.

### `sm_read_mat_file`

`sm_read_mat_file(const char *filename)`

Legacy MAT reader for FloatMatrix (returns NULL on error).

Parameters:

- `filename` (`const char *`): Input MAT file path.

Returns: Pointer to read matrix, or NULL on error. Caller must free with sm_destroy.

### `sm_read_MAT_file`

`sm_read_MAT_file(const char *filename)`

DeprecatedUse sm_read_mat_file instead. filename Input MAT file path. Pointer to read matrix, or NULL on error.

Parameters:

- `filename` (`const char *`): Input MAT file path.

Returns: Pointer to read matrix, or NULL on error.

### `dms_read_mat_file`

`dms_read_mat_file(const char *filename)`

Legacy MAT reader for DoubleSparseMatrix (returns NULL on error).

Parameters:

- `filename` (`const char *`): Input MAT file path.

Returns: Pointer to read sparse matrix, or NULL on error. Caller must free with dms_destroy.

### `dms_read_MAT_file`

`dms_read_MAT_file(const char *filename)`

DeprecatedUse dms_read_mat_file instead. filename Input MAT file path. Pointer to read sparse matrix, or NULL on error.

Parameters:

- `filename` (`const char *`): Input MAT file path.

Returns: Pointer to read sparse matrix, or NULL on error.

### `dm_write_mat_file_ex`

`dm_write_mat_file_ex(const DoubleMatrix *matrix, const char *filename)`

Preferred MAT writer for DoubleMatrix with global format/compression.

Parameters:

- `matrix` (`const DoubleMatrix *`): Matrix to write.
- `filename` (`const char *`): Output file path.

Returns: Status code (MIO_STATUS_OK on success).

### `dm_write_MAT_file_ex`

`dm_write_MAT_file_ex(const DoubleMatrix *matrix, const char *filename)`

DeprecatedUse dm_write_mat_file_ex instead. matrix Matrix to write. filename Output file path. Status code.

Parameters:

- `matrix` (`const DoubleMatrix *`): Matrix to write.
- `filename` (`const char *`): Output file path.

Returns: Status code.

### `sm_write_mat_file_ex`

`sm_write_mat_file_ex(const FloatMatrix *matrix, const char *filename)`

Preferred MAT writer for FloatMatrix.

Parameters:

- `matrix` (`const FloatMatrix *`): Matrix to write.
- `filename` (`const char *`): Output file path.

Returns: Status code (MIO_STATUS_OK on success).

### `sm_write_MAT_file_ex`

`sm_write_MAT_file_ex(const FloatMatrix *matrix, const char *filename)`

DeprecatedUse sm_write_mat_file_ex instead. matrix Matrix to write. filename Output file path. Status code.

Parameters:

- `matrix` (`const FloatMatrix *`): Matrix to write.
- `filename` (`const char *`): Output file path.

Returns: Status code.

### `dm_read_mat_file_ex`

`dm_read_mat_file_ex(const char *filename, DoubleMatrix **out_matrix)`

Preferred MAT reader for DoubleMatrix (out_matrix receives result).

Parameters:

- `filename` (`const char *`): Input MAT file path.
- `out_matrix` (`DoubleMatrix **`): Output matrix pointer (allocated by function, caller must free).

Returns: Status code (MIO_STATUS_OK on success).

### `dm_read_MAT_file_ex`

`dm_read_MAT_file_ex(const char *filename, DoubleMatrix **out_matrix)`

DeprecatedUse dm_read_mat_file_ex instead. filename Input MAT file path. out_matrix Output matrix pointer. Status code.

Parameters:

- `filename` (`const char *`): Input MAT file path.
- `out_matrix` (`DoubleMatrix **`): Output matrix pointer.

Returns: Status code.

### `sm_read_mat_file_ex`

`sm_read_mat_file_ex(const char *filename, FloatMatrix **out_matrix)`

Preferred MAT reader for FloatMatrix (out_matrix receives result).

Parameters:

- `filename` (`const char *`): Input MAT file path.
- `out_matrix` (`FloatMatrix **`): Output matrix pointer (allocated by function, caller must free).

Returns: Status code (MIO_STATUS_OK on success).

### `sm_read_MAT_file_ex`

`sm_read_MAT_file_ex(const char *filename, FloatMatrix **out_matrix)`

DeprecatedUse sm_read_mat_file_ex instead. filename Input MAT file path. out_matrix Output matrix pointer. Status code.

Parameters:

- `filename` (`const char *`): Input MAT file path.
- `out_matrix` (`FloatMatrix **`): Output matrix pointer.

Returns: Status code.

### `dms_read_mat_file_ex`

`dms_read_mat_file_ex(const char *filename, DoubleSparseMatrix **out_matrix)`

Preferred MAT reader for DoubleSparseMatrix (out_matrix receives result).

Parameters:

- `filename` (`const char *`): Input MAT file path.
- `out_matrix` (`DoubleSparseMatrix **`): Output sparse matrix pointer (allocated by function, caller must free).

Returns: Status code (MIO_STATUS_OK on success).

### `dms_read_MAT_file_ex`

`dms_read_MAT_file_ex(const char *filename, DoubleSparseMatrix **out_matrix)`

DeprecatedUse dms_read_mat_file_ex instead. filename Input MAT file path. out_matrix Output sparse matrix pointer. Status code.

Parameters:

- `filename` (`const char *`): Input MAT file path.
- `out_matrix` (`DoubleSparseMatrix **`): Output sparse matrix pointer.

Returns: Status code.

### `dms_read_matrix_market`

`dms_read_matrix_market(const char *filename)`

Read sparse matrix from MatrixMarket file.

Parameters:

- `filename` (`const char *`): Input MatrixMarket file path (*.mtx).

Returns: Sparse matrix pointer (caller must free with dms_destroy), or NULL on error.

### `dms_write_matrix_market`

`void dms_write_matrix_market(const DoubleSparseMatrix *mat, const char *filename)`

Write sparse matrix to MatrixMarket file.

Parameters:

- `mat` (`const DoubleSparseMatrix *`): Sparse matrix to write.
- `filename` (`const char *`): Output MatrixMarket file path (*.mtx).
