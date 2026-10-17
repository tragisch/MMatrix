# `sm_mps.h` – GPU-resident Float Matrix API

## Types

### Typedefs

- `struct SmMpsMatrix`
- `struct SmMpsStream`
- `struct SmMpsGemmPlan`
- `struct SmMpsCounters SmMpsCounters`

## Functions

### `mps_get_shared_device`

`void * mps_get_shared_device(void)`

Returns the shared MTLDevice as an opaque pointer, or NULL if unavailable.

Parameters:

- `(unnamed)` (`void`)

### `mps_get_shared_command_queue`

`void * mps_get_shared_command_queue(void)`

Returns the shared MTLCommandQueue as an opaque pointer, or NULL if unavailable.

Parameters:

- `(unnamed)` (`void`)

### `sm_mps_get_counters`

`sm_mps_get_counters(void)`

Parameters:

- `(unnamed)` (`void`)

### `sm_mps_reset_counters`

`void sm_mps_reset_counters(void)`

Parameters:

- `(unnamed)` (`void`)

### `sm_mps_matrix_create`

`sm_mps_matrix_create(size_t rows, size_t cols)`

Parameters:

- `rows` (`size_t`)
- `cols` (`size_t`)

### `sm_mps_matrix_destroy`

`void sm_mps_matrix_destroy(SmMpsMatrix *matrix)`

Parameters:

- `matrix` (`SmMpsMatrix *`)

### `sm_mps_matrix_contents`

`float * sm_mps_matrix_contents(SmMpsMatrix *matrix)`

Parameters:

- `matrix` (`SmMpsMatrix *`)

### `sm_mps_matrix_const_contents`

`const float * sm_mps_matrix_const_contents(const SmMpsMatrix *matrix)`

Parameters:

- `matrix` (`const SmMpsMatrix *`)

### `sm_mps_matrix_upload`

`bool sm_mps_matrix_upload(SmMpsMatrix *matrix, const float *values)`

Parameters:

- `matrix` (`SmMpsMatrix *`)
- `values` (`const float *`)

### `sm_mps_matrix_download`

`bool sm_mps_matrix_download(const SmMpsMatrix *matrix, float *values)`

Parameters:

- `matrix` (`const SmMpsMatrix *`)
- `values` (`float *`)

### `sm_mps_stream_create`

`sm_mps_stream_create(void)`

Parameters:

- `(unnamed)` (`void`)

### `sm_mps_stream_commit`

`bool sm_mps_stream_commit(SmMpsStream *stream)`

Parameters:

- `stream` (`SmMpsStream *`)

### `sm_mps_stream_wait`

`bool sm_mps_stream_wait(SmMpsStream *stream)`

Parameters:

- `stream` (`SmMpsStream *`)

### `sm_mps_stream_destroy`

`void sm_mps_stream_destroy(SmMpsStream *stream)`

Parameters:

- `stream` (`SmMpsStream *`)

### `sm_mps_gemm_plan_create`

`sm_mps_gemm_plan_create(size_t result_rows, size_t result_cols, size_t interior_cols, bool transpose_left, bool transpose_right, float alpha, float beta)`

Parameters:

- `result_rows` (`size_t`)
- `result_cols` (`size_t`)
- `interior_cols` (`size_t`)
- `transpose_left` (`bool`)
- `transpose_right` (`bool`)
- `alpha` (`float`)
- `beta` (`float`)

### `sm_mps_gemm_plan_destroy`

`void sm_mps_gemm_plan_destroy(SmMpsGemmPlan *plan)`

Parameters:

- `plan` (`SmMpsGemmPlan *`)

### `sm_mps_gemm_plan_encode`

`bool sm_mps_gemm_plan_encode(SmMpsStream *stream, const SmMpsGemmPlan *plan, SmMpsMatrix *C, const SmMpsMatrix *A, const SmMpsMatrix *B)`

Parameters:

- `stream` (`SmMpsStream *`)
- `plan` (`const SmMpsGemmPlan *`)
- `C` (`SmMpsMatrix *`)
- `A` (`const SmMpsMatrix *`)
- `B` (`const SmMpsMatrix *`)

### `sm_mps_matrix_gemm_async`

`bool sm_mps_matrix_gemm_async(SmMpsStream *stream, SmMpsMatrix *C, float alpha, const SmMpsMatrix *A, bool transpose_left, const SmMpsMatrix *B, bool transpose_right, float beta)`

Parameters:

- `stream` (`SmMpsStream *`)
- `C` (`SmMpsMatrix *`)
- `alpha` (`float`)
- `A` (`const SmMpsMatrix *`)
- `transpose_left` (`bool`)
- `B` (`const SmMpsMatrix *`)
- `transpose_right` (`bool`)
- `beta` (`float`)

### `sm_mps_matrix_bias_relu_async`

`bool sm_mps_matrix_bias_relu_async(SmMpsStream *stream, SmMpsMatrix *C, const SmMpsMatrix *bias, bool bias_is_row)`

Parameters:

- `stream` (`SmMpsStream *`)
- `C` (`SmMpsMatrix *`)
- `bias` (`const SmMpsMatrix *`)
- `bias_is_row` (`bool`)

### `sm_mps_matrix_gemm_bias_relu_async`

`bool sm_mps_matrix_gemm_bias_relu_async(SmMpsStream *stream, SmMpsMatrix *C, float alpha, const SmMpsMatrix *A, bool transpose_left, const SmMpsMatrix *B, bool transpose_right, const SmMpsMatrix *bias, bool bias_is_row)`

Parameters:

- `stream` (`SmMpsStream *`)
- `C` (`SmMpsMatrix *`)
- `alpha` (`float`)
- `A` (`const SmMpsMatrix *`)
- `transpose_left` (`bool`)
- `B` (`const SmMpsMatrix *`)
- `transpose_right` (`bool`)
- `bias` (`const SmMpsMatrix *`)
- `bias_is_row` (`bool`)

### `sm_mps_matrix_gemm_bias_relu_ex`

`bool sm_mps_matrix_gemm_bias_relu_ex(SmMpsMatrix *C, float alpha, const SmMpsMatrix *A, bool transpose_left, const SmMpsMatrix *B, bool transpose_right, const SmMpsMatrix *bias, bool bias_is_row)`

Parameters:

- `C` (`SmMpsMatrix *`)
- `alpha` (`float`)
- `A` (`const SmMpsMatrix *`)
- `transpose_left` (`bool`)
- `B` (`const SmMpsMatrix *`)
- `transpose_right` (`bool`)
- `bias` (`const SmMpsMatrix *`)
- `bias_is_row` (`bool`)

### `sm_mps_matrix_gemm_ex`

`bool sm_mps_matrix_gemm_ex(SmMpsMatrix *C, float alpha, const SmMpsMatrix *A, bool transpose_left, const SmMpsMatrix *B, bool transpose_right, float beta)`

Parameters:

- `C` (`SmMpsMatrix *`)
- `alpha` (`float`)
- `A` (`const SmMpsMatrix *`)
- `transpose_left` (`bool`)
- `B` (`const SmMpsMatrix *`)
- `transpose_right` (`bool`)
- `beta` (`float`)

### `mps_matrix_multiply`

`bool mps_matrix_multiply(const float *mat1, size_t rows1, size_t cols1, const float *mat2, size_t rows2, size_t cols2, float *result)`

Parameters:

- `mat1` (`const float *`)
- `rows1` (`size_t`)
- `cols1` (`size_t`)
- `mat2` (`const float *`)
- `rows2` (`size_t`)
- `cols2` (`size_t`)
- `result` (`float *`)

### `mps_matrix_multiply_ex`

`bool mps_matrix_multiply_ex(const float *mat1, size_t rows1, size_t cols1, bool transpose_left, const float *mat2, size_t rows2, size_t cols2, bool transpose_right, float alpha, float beta, float *result, size_t result_rows, size_t result_cols)`

Parameters:

- `mat1` (`const float *`)
- `rows1` (`size_t`)
- `cols1` (`size_t`)
- `transpose_left` (`bool`)
- `mat2` (`const float *`)
- `rows2` (`size_t`)
- `cols2` (`size_t`)
- `transpose_right` (`bool`)
- `alpha` (`float`)
- `beta` (`float`)
- `result` (`float *`)
- `result_rows` (`size_t`)
- `result_cols` (`size_t`)
