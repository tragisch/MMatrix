/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "sm.h"
#include <omp.h>
#include <pcg_variants.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
// Prevent automatic use of bfloat16 types in NEON vector operations
#ifdef __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#undef __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#endif
#ifdef __ARM_FEATURE_BF16_SCALAR_ARITHMETIC
#undef __ARM_FEATURE_BF16_SCALAR_ARITHMETIC
#endif
#endif

#define INIT_CAPACITY 100
static const float EPSILON = 1e-5f;

/*******************************/
/*      Define Environment     */
/*******************************/

#if defined(USE_ACCELERATE)
#define ACTIVE_LIB "Apple Accelerate"
#elif defined(USE_ACCELERATE_MPS)
#define ACTIVE_LIB "Metal Performance Shaders"
#elif defined(USE_OPENBLAS)
#define ACTIVE_LIB "OpenBLAS"
#else
#if defined(__ARM_NEON)
#define ACTIVE_LIB "OpenMP or ARM NEON"
#else
#define ACTIVE_LIB "No BLAS"
#endif
#endif

#if defined(USE_ACCELERATE)
#define BLASINT int
#include <Accelerate/Accelerate.h>
#elif defined(USE_ACCELERATE_MPS)
#define BLASINT int
#include "sm_mps.h"
#include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
#define BLASINT int
#include <cblas.h>
#include <lapacke.h>
#endif

// Block size used for cache-optimized transpose operations
#define BLOCK_SIZE 64

/*******************************/
/*       Private Functions     */
/*******************************/

size_t sm_rank_euler(const FloatMatrix *mat) {
  size_t rows = mat->rows;
  size_t cols = mat->cols;
  FloatMatrix *copy = sm_create(rows, cols);
  if (!copy) {
    perror("Error: Memory allocation for matrix copy failed.\n");
    return 0;
  }
  memcpy(copy->values, mat->values, rows * cols * sizeof(float));

  // Inline LU elimination (no pivot matrix, just elimination)
  FloatMatrix *dummy = copy;
  size_t n = dummy->rows;
  for (size_t pivot = 0; pivot < n; pivot++) {
    float pivot_val = dummy->values[pivot * dummy->cols + pivot];
    if (fabs(pivot_val) < EPSILON)
      continue;
    for (size_t row = pivot + 1; row < n; row++) {
      float factor = dummy->values[row * dummy->cols + pivot] / pivot_val;
      dummy->values[row * dummy->cols + pivot] = 0.0f;
      for (size_t col = pivot + 1; col < dummy->cols; col++) {
        dummy->values[row * dummy->cols + col] -=
            factor * dummy->values[pivot * dummy->cols + col];
      }
    }
  }

  size_t rank = 0;
#pragma omp parallel for reduction(+ : rank)
  for (size_t i = 0; i < rows; i++) {
    size_t found = 0;
    for (size_t j = 0; j < cols; j++) {
      if (fabs(copy->values[i * cols + j]) > EPSILON) {
        found = 1;
      }
    }
    rank += found;
  }

  sm_destroy(copy);
  return rank;
}

static unsigned int sm_random_seed(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (unsigned int)(ts.tv_nsec ^ ts.tv_sec);
}

float *sm_to_column_major(const FloatMatrix *mat) {
  size_t rows = mat->rows;
  size_t cols = mat->cols;
  float *col_major = malloc(rows * cols * sizeof(float));
  if (!col_major) {
    perror("Failed to allocate column-major buffer");
    return NULL;
  }

#pragma omp parallel for
  for (size_t j = 0; j < cols; ++j) {
#pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
      col_major[j * rows + i] = mat->values[i * cols + j];
    }
  }
  return col_major;
}

/*******************************/
/*      Public Functions      */
/*******************************/

const char *sm_active_library(void) { return ACTIVE_LIB; }

FloatMatrix *sm_create_empty(void) {
  FloatMatrix *matrix = (FloatMatrix *)malloc(sizeof(FloatMatrix));
  matrix->rows = 0;
  matrix->cols = 0;
  matrix->capacity = 0;
  matrix->values = NULL;
  return matrix;
}

FloatMatrix *sm_create_zeros(size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *matrix = (FloatMatrix *)malloc(sizeof(FloatMatrix));
  if (!matrix) {
    perror("Error: could not allocate FloatMatrix struct.\n");
    return NULL;
  }
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->capacity = rows * cols;
  matrix->values = (float *)calloc(rows * cols, sizeof(float));
  if (!matrix->values) {
    free(matrix);
    perror("Error: could not allocate values array.\n");
    return NULL;
  }
  return matrix;
}

FloatMatrix *sm_create_with_values(size_t rows, size_t cols, float *values) {
  FloatMatrix *matrix = sm_create(rows, cols);
  if (!matrix)
    return NULL;
  memcpy(matrix->values, values, rows * cols * sizeof(float));
  return matrix;
}

FloatMatrix *sm_create(size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *matrix = (FloatMatrix *)malloc(sizeof(FloatMatrix));
  if (!matrix) {
    perror("Error: could not allocate FloatMatrix struct.\n");
    return NULL;
  }
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->capacity = rows * cols;
  matrix->values = (float *)calloc(rows * cols, sizeof(float));
  if (!matrix->values) {
    free(matrix);
    perror("Error: could not allocate values array.\n");
    return NULL;
  }
  return matrix;
}

FloatMatrix *sm_clone(const FloatMatrix *mat) {
  FloatMatrix *copy = sm_create(mat->rows, mat->cols);
  if (!copy)
    return NULL;
  memcpy(copy->values, mat->values, mat->rows * mat->cols * sizeof(float));
  return copy;
}

FloatMatrix *sm_create_identity(size_t n) {
  FloatMatrix *identity = sm_create(n, n);
  if (!identity)
    return NULL;

  for (size_t i = 0; i < n; i++) {
    identity->values[i * n + i] = 1.0f;
  }
  return identity;
}

FloatMatrix *sm_create_random(size_t rows, size_t cols) {
  if (cols != 0 && rows > SIZE_MAX / cols) {
    perror("Overflow detected in matrix allocation.");
    return NULL;
  }

  FloatMatrix *mat = sm_create(rows, cols);
  size_t size = rows * cols;

  unsigned int global_seed =
      sm_random_seed() ^ (unsigned int)((uintptr_t)mat & 0xFFFFFFFFu);
#pragma omp parallel
  {
    pcg32_random_t rng;
    int thread_id = omp_get_thread_num();
    pcg32_srandom_r(&rng, global_seed ^ (unsigned int)thread_id,
                    (unsigned int)thread_id);
#pragma omp for
    for (size_t i = 0; i < size; ++i) {
      mat->values[i] = (float)pcg32_random_r(&rng) / UINT32_MAX;
    }
  }

  return mat;
}

// He initialization (He-et-al.) random matrix creation
FloatMatrix *sm_create_random_he(size_t rows, size_t cols, size_t fan_in) {
  if (cols != 0 && rows > SIZE_MAX / cols) {
    perror("Overflow detected in matrix allocation.");
    return NULL;
  }

  FloatMatrix *mat = sm_create(rows, cols);
  if (!mat)
    return NULL;

  float stddev = sqrtf(2.0f / fan_in);
  size_t size = rows * cols;

  unsigned int global_seed =
      sm_random_seed() ^ (unsigned int)((uintptr_t)mat & 0xFFFFFFFFu);
#pragma omp parallel
  {
    pcg32_random_t rng;
    int thread_id = omp_get_thread_num();
    pcg32_srandom_r(&rng, global_seed ^ (unsigned int)thread_id,
                    (unsigned int)thread_id);
#pragma omp for
    for (size_t i = 0; i < size; ++i) {
      // Box-Muller Transform (approximate standard normal)
      float u1 = (float)pcg32_random_r(&rng) / UINT32_MAX;
      float u2 = (float)pcg32_random_r(&rng) / UINT32_MAX;
      float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
      mat->values[i] = z * stddev;
    }
  }

  return mat;
}

// Xavier (Glorot) initialization: Normal distribution
FloatMatrix *sm_create_random_xavier(size_t rows, size_t cols, size_t fan_in,
                                     size_t fan_out) {

  if (cols != 0 && rows > SIZE_MAX / cols) {
    perror("Overflow detected in matrix allocation.");
    return NULL;
  }

  FloatMatrix *mat = sm_create(rows, cols);
  if (!mat)
    return NULL;

  float stddev = sqrtf(2.0f / (fan_in + fan_out));
  size_t size = rows * cols;

  unsigned int global_seed =
      sm_random_seed() ^ (unsigned int)((uintptr_t)mat & 0xFFFFFFFFu);
#pragma omp parallel
  {
    pcg32_random_t rng;
    int thread_id = omp_get_thread_num();
    pcg32_srandom_r(&rng, global_seed ^ (unsigned int)thread_id,
                    (unsigned int)thread_id);
#pragma omp for
    for (size_t i = 0; i < size; ++i) {
      // Box-Muller Transform: generate standard normal distributed value
      float u1 = (float)pcg32_random_r(&rng) / UINT32_MAX;
      float u2 = (float)pcg32_random_r(&rng) / UINT32_MAX;
      float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
      mat->values[i] = z * stddev;
    }
  }

  return mat;
}

FloatMatrix *sm_from_array_ptrs(size_t rows, size_t cols, float **array) {
  // check if the array is NULL
  if (array == NULL) {
    perror("Error: array is NULL.\n");
    return NULL;
  }

  FloatMatrix *mat = sm_create(rows, cols);
  if (!mat)
    return NULL;

#pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat->values[i * cols + j] = array[i][j];
    }
  }
  return mat;
}

FloatMatrix *sm_from_array_static(size_t rows, size_t cols,
                                  float array[rows][cols]) {
  // check if the array is NULL
  if (array == NULL) {
    perror("Error: array is NULL.\n");
    return NULL;
  }

  FloatMatrix *matrix = sm_create(rows, cols);
  if (!matrix)
    return NULL;

#pragma omp parallel for
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      matrix->values[i * cols + j] = array[i][j];
    }
  }
  return matrix;
}

float *sm_create_array_from_matrix(FloatMatrix *matrix) {
  if (matrix == NULL || matrix->values == NULL) {
    perror("Error: matrix is NULL.\n");
    return NULL;
  }

  float *array = (float *)malloc(matrix->rows * matrix->cols * sizeof(float));
  if (!array) {
    perror("Error: could not allocate array.\n");
    return NULL;
  }

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < matrix->rows; ++i) {
    for (size_t j = 0; j < matrix->cols; ++j) {
      array[i * matrix->cols + j] = (float)matrix->values[i * matrix->cols + j];
    }
  }
  return array;
}

FloatMatrix *sm_get_row(const FloatMatrix *mat, size_t i) {
  FloatMatrix *row = sm_create(1, mat->cols);
  if (!row)
    return NULL;
  memcpy(row->values, &mat->values[i * mat->cols], mat->cols * sizeof(float));
  return row;
}

FloatMatrix *sm_get_last_row(const FloatMatrix *mat) {
  return sm_get_row(mat, mat->rows - 1);
}

FloatMatrix *sm_get_col(const FloatMatrix *mat, size_t j) {
  FloatMatrix *col = sm_create(mat->rows, 1);
  if (!col)
    return NULL;
  for (size_t i = 0; i < mat->rows; i++) {
    col->values[i] = mat->values[i * mat->cols + j];
  }
  return col;
}

FloatMatrix *sm_get_last_col(const FloatMatrix *mat) {
  return sm_get_col(mat, mat->cols - 1);
}

FloatMatrix *sm_multiply(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (mat1->cols != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *product = sm_create(mat1->rows, mat2->cols);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (BLASINT)mat1->rows,
              (BLASINT)mat2->cols, (BLASINT)mat1->cols, 1.0, mat1->values,
              (BLASINT)mat1->cols, mat2->values, (BLASINT)mat2->cols, 0.0,
              product->values, (BLASINT)product->cols);

#elif defined(USE_ACCELERATE_MPS)
  if (mat1->cols < 2000) {
    vDSP_mmul(mat1->values, 1, mat2->values, 1, product->values, 1, mat1->rows,
              mat2->cols, mat1->cols);
  } else {
    mps_matrix_multiply(mat1->values, mat1->rows, mat1->cols, mat2->values,
                        mat2->rows, mat2->cols, product->values);
  }

#else
  FloatMatrix *mat2_transposed = sm_transpose(mat2);

  size_t rows = mat1->rows;
  size_t cols = mat2->cols;
  size_t inner = mat1->cols;
  float *a = mat1->values;
  float *bT = mat2_transposed->values;
  float *c = product->values;

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
      for (size_t k = 0; k < inner; k++) {
        sum += a[i * inner + k] * bT[j * inner + k];
      }
      c[i * cols + j] = sum;
    }
  }
  sm_destroy(mat2_transposed);

#endif
  return product;
}

// Cache-efficient matrix multiplication using transposed B and dot products
// (like mat_mul4)
FloatMatrix *sm_multiply_4(const FloatMatrix *A, const FloatMatrix *B) {
  if (!A || !B || A->cols != B->rows) {
    perror("Error: invalid matrix dimensions for sm_multiply_4.");
    return NULL;
  }

  size_t n_a_rows = A->rows;
  size_t n_a_cols = A->cols;
  size_t n_b_cols = B->cols;

  FloatMatrix *product = sm_create(n_a_rows, n_b_cols);
  if (!product)
    return NULL;

  FloatMatrix *B_T;
  if (B->rows == B->cols) {
    B_T = sm_clone(B);
    sm_inplace_square_transpose(B_T);
  } else {
    B_T = sm_transpose(B);
  }

  if (!B_T) {
    sm_destroy(product);
    return NULL;
  }

#pragma omp parallel for collapse(2) if (n_a_rows > 500 || n_b_cols > 500)
  for (size_t i = 0; i < n_a_rows; ++i) {
    for (size_t j = 0; j < n_b_cols; ++j) {
      float *row_a = &A->values[i * n_a_cols];
      float *row_bT = &B_T->values[j * n_a_cols];
      float sum = 0.0f;
      for (size_t k = 0; k < n_a_cols; ++k) {
        sum += row_a[k] * row_bT[k];
      }
      product->values[i * n_b_cols + j] = sum;
    }
  }

  sm_destroy(B_T);
  return product;
}

void sm_inplace_elementwise_multiply(FloatMatrix *mat1,
                                     const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !sm_is_equal_size(mat1, mat2)) {
    perror("Error: invalid matrix dimensions for Hadamard product.\n");
    return;
  }

  size_t size = mat1->rows * mat1->cols;

#if defined(USE_ACCELERATE_MPS) || defined(USE_ACCELERATE)

  vDSP_vmul(mat1->values, 1, mat2->values, 1, mat1->values, 1, size);
  return;
#else

  float *a = mat1->values;
  float *b = mat2->values;

#ifdef __ARM_NEON
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t va = vld1q_f32(&a[i]);
    float32x4_t vb = vld1q_f32(&b[i]);
    float32x4_t vc = vmulq_f32(va, vb);
    vst1q_f32(&a[i], vc);
  }
  for (; i < size; i++) {
    a[i] *= b[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    a[i] *= b[i];
  }
#endif
#endif
}

FloatMatrix *sm_elementwise_multiply(const FloatMatrix *mat1,
                                     const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !sm_is_equal_size(mat1, mat2)) {
    perror("Error: invalid matrix dimensions for Hadamard product.\n");
    return NULL;
  }

  FloatMatrix *result = sm_clone(mat1);
  if (!result)
    return NULL;

  sm_inplace_elementwise_multiply(result, mat2);
  return result;
}

FloatMatrix *sm_multiply_by_number(const FloatMatrix *mat, const float number) {
  FloatMatrix *product = sm_clone(mat);
  sm_inplace_multiply_by_number(product, number);
  return product;
}

FloatMatrix *sm_transpose(const FloatMatrix *mat) {
  if (mat == NULL || mat->values == NULL)
    return NULL;

  if (sm_is_square(mat)) {
    FloatMatrix *copy = sm_clone(mat);
    sm_inplace_square_transpose(copy);
    return copy;
  }

  size_t n = mat->rows;
  size_t m = mat->cols;

  FloatMatrix *transposed = sm_create(m, n);

  float *src = mat->values;
  float *dst = transposed->values;

#pragma omp parallel for collapse(2) schedule(dynamic) if (n > 500 || m > 500)
  for (size_t ii = 0; ii < mat->rows; ii += BLOCK_SIZE) {
    for (size_t jj = 0; jj < mat->cols; jj += BLOCK_SIZE) {
      for (size_t i = ii; i < ii + BLOCK_SIZE && i < mat->rows; i++) {
        for (size_t j = jj; j < jj + BLOCK_SIZE && j < mat->cols; j++) {
          dst[j * mat->rows + i] = src[i * mat->cols + j];
        }
      }
    }
  }

  return transposed;
}

FloatMatrix *sm_solve_system(const FloatMatrix *A, const FloatMatrix *b) {
  if (!A || !b || A->rows != A->cols || A->rows != b->rows) {
    perror("Error: invalid matrix dimensions for solve.\n");
    return NULL;
  }
#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
  int n = (int)A->rows;
  int nrhs = (int)b->cols;
  int lda = n;
  int ldb = n;

  FloatMatrix *a_copy = sm_clone(A);
  FloatMatrix *b_copy = sm_clone(b);

  float *a_col = sm_to_column_major(a_copy);
  float *b_col = sm_to_column_major(b_copy);

  int *ipiv = malloc((size_t)n * sizeof(int));
  int info;

  // Solve AX = B using LAPACK
  sgesv_(&n, &nrhs, a_col, &lda, ipiv, b_col, &ldb, &info);

  free(ipiv);
  free(a_col);
  sm_destroy(a_copy);

  if (info != 0) {
    fprintf(stderr, "Error: sgesv_ failed with info = %d\n", info);
    sm_destroy(b_copy);
    free(b_col);
    return NULL;
  }

  float *restrict dst = b_copy->values;
  float *restrict src = b_col;
  size_t rows = b_copy->rows;
  size_t cols = b_copy->cols;

#pragma omp parallel
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      dst[i * cols + j] = src[j * rows + i];
    }
  }

  free(b_col);
  return b_copy;

#elif defined(USE_OPENBLAS)

  int n = (int)A->rows;
  int nrhs = (int)b->cols;
  int lda = (int)A->cols;
  int ldb = (int)b->cols;

  FloatMatrix *a_copy = sm_clone(A);
  FloatMatrix *b_copy = sm_clone(b);

  int *ipiv = malloc((size_t)n * sizeof(int));
  int info;

  info = LAPACKE_sgesv(LAPACK_ROW_MAJOR,
                       n,    // number of equations
                       nrhs, // number of right-hand sides
                       a_copy->values,
                       lda, // leading dimension of A (= cols in row-major)
                       ipiv, b_copy->values,
                       ldb // leading dimension of B (= cols in row-major)
  );

  free(ipiv);

  if (info != 0) {
    fprintf(stderr, "Error: LAPACKE_sgesv failed with info = %d\n", info);
    sm_destroy(a_copy);
    sm_destroy(b_copy);
    return NULL;
  }

  sm_destroy(a_copy);
  return b_copy;
#else
  size_t n = A->rows;
  size_t rhs = b->cols;

  FloatMatrix *lu = sm_clone(A);
  FloatMatrix *x = sm_clone(b);
  size_t *pivot_order = (size_t *)malloc(n * sizeof(size_t));

  if (!sm_lu_decompose(lu, pivot_order)) {
    perror("Error: LU decomposition failed.\n");
    sm_destroy(lu);
    sm_destroy(x);
    free(pivot_order);
    return NULL;
  }

  for (size_t i = 0; i < n; ++i) {
    if (pivot_order[i] != i) {
      for (size_t j = 0; j < rhs; ++j) {
        float tmp = x->values[i * rhs + j];
        x->values[i * rhs + j] = x->values[pivot_order[i] * rhs + j];
        x->values[pivot_order[i] * rhs + j] = tmp;
      }
    }
  }
#pragma omp parallel for
  for (size_t k = 0; k < rhs; ++k) {
    for (size_t i = 1; i < n; ++i) {
      float sum = x->values[i * rhs + k];
      for (size_t j = 0; j < i; ++j) {
        sum -= lu->values[i * n + j] * x->values[j * rhs + k];
      }
      x->values[i * rhs + k] = sum;
    }

    size_t i = n;
    do {
      --i;
      float sum = x->values[i * rhs + k];
      for (size_t j = i + 1; j < n; ++j) {
        sum -= lu->values[i * n + j] * x->values[j * rhs + k];
      }
      x->values[i * rhs + k] = sum / lu->values[i * n + i];
    } while (i != 0);
  }

  sm_destroy(lu);
  free(pivot_order);
  return x;
#endif
}

bool sm_is_equal(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (mat1 == NULL || mat2 == NULL) {
    return false;
  }
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    return false;
  }

  size_t size = mat1->cols * mat1->rows;
  int equal = 1;
#ifdef __ARM_NEON
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t a = vld1q_f32(&mat1->values[i]);
    float32x4_t b = vld1q_f32(&mat2->values[i]);
    float32x4_t diff = vabsq_f32(vsubq_f32(a, b));
    float32x4_t eps = vdupq_n_f32(EPSILON);
    uint32x4_t cmp = vcgtq_f32(diff, eps);
    if (vmaxvq_u32(cmp) != 0) {
      equal = 0;
      break;
    }
  }
  for (; i < size; i++) {
    if (fabsf(mat1->values[i] - mat2->values[i]) > EPSILON) {
      equal = 0;
      break;
    }
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    if (fabs(mat1->values[i] - mat2->values[i]) > EPSILON) {
#pragma omp atomic write
      equal = 0;
    }
  }
#endif
  return equal;
}

FloatMatrix *sm_add(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *sum = sm_clone(mat1);
  sm_inplace_add(sum, mat2);
  return sum;
}

FloatMatrix *sm_diff(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *difference = sm_clone(mat1);
  sm_inplace_diff(difference, mat2);
  return difference;
}

// Use TOLERANCE for numerical zero threshold in LU decomposition
#define TOLERANCE EPSILON
bool sm_lu_decompose(FloatMatrix *mat, size_t *pivot_order) {
  size_t n = mat->rows;
  if (mat->cols != n)
    return false;

  for (size_t pivot = 0; pivot < n; pivot++) {
    float max_val = (float)fabs(mat->values[pivot * n + pivot]);
    size_t max_row = pivot;
    for (size_t row = pivot + 1; row < n; row++) {
      float val = (float)fabs(mat->values[row * n + pivot]);
      if (val > max_val) {
        max_val = val;
        max_row = row;
      }
    }

    // Use TOLERANCE instead of magic value
    if (max_val < TOLERANCE) {
      return false;
    }

    pivot_order[pivot] = max_row;

    if (max_row != pivot) {
      for (size_t col = 0; col < n; col++) {
        float tmp = mat->values[pivot * n + col];
        mat->values[pivot * n + col] = mat->values[max_row * n + col];
        mat->values[max_row * n + col] = tmp;
      }
    }

    float pivot_val = mat->values[pivot * n + pivot];
#pragma omp parallel for
    for (size_t row = pivot + 1; row < n; row++) {
      float factor = mat->values[row * n + pivot] / pivot_val;
      mat->values[row * n + pivot] = factor;
      for (size_t col = pivot + 1; col < n; col++) {
        mat->values[row * n + col] -= factor * mat->values[pivot * n + col];
      }
    }
  }

  return true;
}

float sm_determinant(const FloatMatrix *mat) {
  if (mat->cols != mat->rows) {
    perror("the Matrix has to be square!");
    return 0.0f;
  }
  if (mat->cols == 1) {
    return sm_get(mat, 0, 0);
  } else if (mat->cols == 2) {
    return sm_get(mat, 0, 0) * sm_get(mat, 1, 1) -
           sm_get(mat, 0, 1) * sm_get(mat, 1, 0);
  } else if (mat->cols == 3) {
    float *a = mat->values;
    float det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
                a[1] * (a[3] * a[8] - a[5] * a[6]) +
                a[2] * (a[3] * a[7] - a[4] * a[6]);
    return det;
  } else {

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) ||                        \
    defined(USE_ACCELERATE_MPS)

    BLASINT *ipiv = (BLASINT *)malloc(mat->cols * sizeof(BLASINT));
    FloatMatrix *lu = sm_clone(mat);
    BLASINT info = 0;
    BLASINT cols = (BLASINT)lu->cols;
    BLASINT rows = (BLASINT)lu->rows;

    sgetrf_(&cols, &rows, lu->values, &cols, ipiv, &info);
    if (info != 0) {
      perror("Error: dgetrf failed.\n");
      free(ipiv);
      sm_destroy(lu);
      return 0;
    }
    float det = 1.0;
    for (size_t i = 0; i < mat->cols; i++) {
      det *= sm_get(lu, i, i);
    }
    free(ipiv);
    sm_destroy(lu);

#else
    FloatMatrix *copy = sm_clone(mat);
    if (!copy)
      return 0.0f;

    size_t n = mat->rows;
    float *a = copy->values;

    for (size_t k = 0; k < n; ++k) {
      // Pivot
      size_t max_row = k;
      float max_val = fabsf(a[k * n + k]);
      for (size_t i = k + 1; i < n; ++i) {
        float val = fabsf(a[i * n + k]);
        if (val > max_val) {
          max_val = val;
          max_row = i;
        }
      }

      if (max_val < 1e-6f) {
        sm_destroy(copy);
        return 0.0f;
      }

      if (max_row != k) {
        for (size_t j = 0; j < n; ++j) {
          float tmp = a[k * n + j];
          a[k * n + j] = a[max_row * n + j];
          a[max_row * n + j] = tmp;
        }
      }

      float pivot = a[k * n + k];
      for (size_t i = k + 1; i < n; ++i) {
        float factor = a[i * n + k] / pivot;
        a[i * n + k] = 0.0f;
        for (size_t j = k + 1; j < n; ++j) {
          a[i * n + j] -= factor * a[k * n + j];
        }
      }
    }

    float det = 1.0f;
    for (size_t i = 0; i < n; ++i) {
      det *= a[i * n + i];
    }

    sm_destroy(copy);
#endif
    return det;
  }
}

FloatMatrix *sm_inverse(const FloatMatrix *mat) {
  if (mat->cols != mat->rows || mat->rows == 0 || mat->cols == 0) {
    perror("the Matrix has to be square!");
    return NULL;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) ||                        \
    defined(USE_ACCELERATE_MPS)
  FloatMatrix *inverse = sm_clone(mat);
  BLASINT *ipiv = (BLASINT *)malloc(mat->cols * sizeof(BLASINT));
  if (ipiv == NULL) {
    free(inverse->values);
    free(inverse);
    perror("Error: Memory allocation for ipiv failed.\n");
    return NULL;
  }
  BLASINT info = 0;
  BLASINT n = (BLASINT)inverse->cols;

  sgetrf_(&n, &n, inverse->values, &n, ipiv, &info);
  if (info != 0) {
    free(ipiv);
    free(inverse->values);
    free(inverse);
    perror("Error: dgetrf failed.\n");
    return NULL;
  }

  BLASINT lwork = -1;
  float work_opt;
  sgetri_(&n, inverse->values, &n, ipiv, &work_opt, &lwork, &info);

  lwork = (BLASINT)work_opt;
  float *work = (float *)malloc((size_t)lwork * sizeof(float));
  if (work == NULL) {
    free(ipiv);
    free(inverse->values);
    free(inverse);
    perror("Error: Memory allocation for work array failed.\n");
    return NULL;
  }

  sgetri_(&n, inverse->values, &n, ipiv, work, &lwork, &info);
  free(work);
  free(ipiv);
  if (info != 0) {
    free(inverse->values);
    free(inverse);
    perror("Error: dgetri failed.\n");
    return NULL;
  }
  return inverse;
#else
  // Neue Nicht-BLAS-Variante: LU-Zerlegung und
  // Vorwärts-/Rückwärtssubstitution
  if (!sm_is_square(mat)) {
    perror("Error: Matrix must be square.\n");
    return NULL;
  }

  size_t n = mat->cols;
  FloatMatrix *copy = sm_clone(mat);
  if (!copy)
    return NULL;

  FloatMatrix *inverse = sm_create_identity(n);
  if (!inverse) {
    sm_destroy(copy);
    return NULL;
  }

  size_t *pivot_order = (size_t *)malloc(n * sizeof(size_t));
  if (!pivot_order) {
    sm_destroy(copy);
    sm_destroy(inverse);
    return NULL;
  }

  if (!sm_lu_decompose(copy, pivot_order)) {
    perror("Error: LU decomposition failed.\n");
    free(pivot_order);
    sm_destroy(copy);
    sm_destroy(inverse);
    return NULL;
  }

  for (size_t col = 0; col < n; col++) {
    // Vorwärtseinsetzen (L * y = e_col)
    for (size_t i = 1; i < n; i++) {
      float sum = inverse->values[i * n + col];
      for (size_t j = 0; j < i; j++) {
        sum -= copy->values[i * n + j] * inverse->values[j * n + col];
      }
      inverse->values[i * n + col] = sum;
    }

    // Rückwärtseinsetzen (U * x = y)
    size_t i = n;
    do {
      --i;
      float sum = inverse->values[i * n + col];
      for (size_t j = i + 1; j < n; j++) {
        sum -= copy->values[i * n + j] * inverse->values[j * n + col];
      }
      inverse->values[i * n + col] = sum / copy->values[i * n + i];
    } while (i != 0);
  }

  free(pivot_order);
  sm_destroy(copy);
  return inverse;
#endif
}

void sm_set(FloatMatrix *mat, size_t i, size_t j, const float value) {
  mat->values[i * mat->cols + j] = value;
}

float sm_get(const FloatMatrix *mat, size_t i, size_t j) {
  return mat->values[i * mat->cols + j];
}

void sm_reshape(FloatMatrix *matrix, size_t new_rows, size_t new_cols) {
  matrix->rows = new_rows;
  matrix->cols = new_cols;
}

void sm_resize(FloatMatrix *mat, size_t new_row, size_t new_col) {
  // allocate new memory for dense matrix:
  float *new_values = (float *)calloc(new_row * new_col, sizeof(float));
  if (new_values == NULL) {
    perror("Error: could not reallocate memory for dense matrix.\n");
    exit(EXIT_FAILURE);
  }

  // Copy values from old matrix to new matrix using linear index arithmetic
  // and OpenMP
  size_t min_rows = (new_row < mat->rows) ? new_row : mat->rows;
  size_t min_cols = (new_col < mat->cols) ? new_col : mat->cols;
  size_t n = min_rows * min_cols;
#pragma omp parallel for
  for (size_t idx = 0; idx < n; ++idx) {
    size_t i = idx / min_cols;
    size_t j = idx % min_cols;
    new_values[i * new_col + j] = mat->values[i * mat->cols + j];
  }

  free(mat->values);
  mat->values = new_values;
  mat->rows = new_row;
  mat->cols = new_col;
  mat->capacity = new_row * new_col;
}

// Slices rows [start, end) from mat and returns a new FloatMatrix
FloatMatrix *sm_slice_rows(const FloatMatrix *mat, size_t start, size_t end) {
  if (!mat || start >= end || end > mat->rows) {
    return NULL;
  }

  size_t num_rows = end - start;
  size_t cols = mat->cols;
  FloatMatrix *slice = sm_create(num_rows, cols);
  if (!slice)
    return NULL;

  float *dst = slice->values;
  const float *src = mat->values + start * cols;

#if defined(__ARM_NEON)
  size_t total = num_rows * cols;
  size_t i = 0;
  for (; i + 4 <= total; i += 4) {
    vst1q_f32(&dst[i], vld1q_f32(&src[i]));
  }
  for (; i < total; ++i) {
    dst[i] = src[i];
  }

#else
#pragma omp parallel for
  for (size_t i = 0; i < num_rows; ++i) {
    memcpy(&dst[i * cols], &src[i * cols], cols * sizeof(float));
  }
#endif

  return slice;
}

void sm_print(const FloatMatrix *matrix) {
  for (size_t i = 0; i < matrix->rows; i++) {
    printf("[ ");
    for (size_t j = 0; j < matrix->cols; j++) {
      printf("%.2f ", sm_get(matrix, i, j));
    }
    printf("]\n");
  }
  printf("Matrix (%zu x %zu)\n", matrix->rows, matrix->cols);
}

void sm_destroy(FloatMatrix *mat) {
  if (!mat)
    return;
  if (mat->values) {
    free(mat->values);
  }
  free(mat);
}

float sm_trace(const FloatMatrix *mat) {
  size_t n = (mat->rows < mat->cols) ? mat->rows : mat->cols;
  float trace = 0;
  for (size_t i = 0; i < n; i++) {
    trace += sm_get(mat, i, i);
  }
  return trace;
}

float sm_norm(const FloatMatrix *mat) {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) ||                        \
    defined(USE_ACCELERATE_MPS)
  return cblas_snrm2((BLASINT)(mat->rows * mat->cols), mat->values, 1);
#else
#ifdef __ARM_NEON
  size_t n = mat->rows * mat->cols;
  size_t i = 0;
  float32x4_t sum_vec = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(&mat->values[i]);
    sum_vec = vmlaq_f32(sum_vec, v, v);
  }

  float buf[4];
  vst1q_f32(buf, sum_vec);
  float norm = buf[0] + buf[1] + buf[2] + buf[3];

  for (; i < n; ++i) {
    norm += mat->values[i] * mat->values[i];
  }
  return sqrtf(norm);
#else
  float norm = 0;
  for (size_t i = 0; i < mat->rows * mat->cols; i++) {
    norm += mat->values[i] * mat->values[i];
  }
  return sqrtf(norm);
#endif
#endif
}

size_t sm_rank(const FloatMatrix *mat) {
  if (mat == NULL || mat->values == NULL) {
    return 0; // No matrix, no rank
  }
  size_t rank = 0;
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) ||                        \
    defined(USE_ACCELERATE_MPS)
  BLASINT m = (BLASINT)mat->rows;
  BLASINT n = (BLASINT)mat->cols;
  BLASINT lda = n;
  BLASINT lwork = -1;
  float wkopt;
  float *work;
  BLASINT info;

  sgeqrf_(&m, &n, mat->values, &lda, NULL, &wkopt, &lwork, &info);
  lwork = (BLASINT)wkopt;
  work = (float *)malloc((size_t)lwork * sizeof(float));
  if (work == NULL) {
    return 0; // Memory allocation failed
  }

  float *tau = (float *)malloc((size_t)(m < n ? m : n) * sizeof(float));
  if (tau == NULL) {
    free(work);
    return 0; // Memory allocation failed
  }

  sgeqrf_(&m, &n, mat->values, &lda, tau, work, &lwork, &info);
  free(work);
  free(tau);
  if (info != 0) {
    return 0; // QR factorization failed
  }

  int k = (m < n) ? m : n;
  for (int i = 0; i < k; ++i) {
    if (fabs(mat->values[i * lda + i]) > EPSILON) {
      rank++;
    }
  }
#else
  rank = sm_rank_euler(mat);
#endif
  return rank;
}

float sm_density(const FloatMatrix *mat) {
  size_t size = mat->rows * mat->cols;
  int counter = 0;
#ifdef __ARM_NEON
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t v = vld1q_f32(&mat->values[i]);
    float32x4_t abs_v = vabsq_f32(v);
    float32x4_t eps = vdupq_n_f32(EPSILON);
    uint32x4_t cmp = vcgtq_f32(abs_v, eps);
    counter += (int)vaddvq_u32(cmp);
  }
  for (; i < size; ++i) {
    if (fabsf(mat->values[i]) > EPSILON) {
      counter++;
    }
  }
#else
#pragma omp parallel for reduction(+ : counter)
  for (size_t i = 0; i < size; i++) {
    if (fabs(mat->values[i]) > EPSILON) {
      counter++;
    }
  }
#endif
  return (float)counter / (float)size;
}

// Matrix is empty
bool sm_is_empty(const FloatMatrix *mat) {
  return (mat == NULL || mat->values == NULL || mat->rows == 0 ||
          mat->cols == 0);
}

// Matrix is square
bool sm_is_square(const FloatMatrix *mat) { return (mat->rows == mat->cols); }

// Matrix is vector
bool sm_is_vector(const FloatMatrix *mat) {
  return (mat->rows == 1 || mat->cols == 1);
}

// Matrix is equal size
bool sm_is_equal_size(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  return (mat1->rows == mat2->rows && mat1->cols == mat2->cols);
}

// In-place operations
void sm_inplace_add(FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
    perror("Error: invalid matrix dimensions.\n");
    return;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) ||                        \
    defined(USE_ACCELERATE_MPS)
  cblas_saxpy((BLASINT)(mat1->rows * mat1->cols), 1.0, mat2->values, 1,
              mat1->values, 1);
#else
#ifdef __ARM_NEON
  size_t n = mat1->rows * mat1->cols;
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t a = vld1q_f32(&mat1->values[i]);
    float32x4_t b = vld1q_f32(&mat2->values[i]);
    vst1q_f32(&mat1->values[i], vaddq_f32(a, b));
  }
  for (; i < n; i++) {
    mat1->values[i] += mat2->values[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < mat1->rows * mat1->cols; i++) {
    mat1->values[i] += mat2->values[i];
  }
#endif
#endif
}

// In-place difference
void sm_inplace_diff(FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
    perror("Error: invalid matrix dimensions.\n");
    return;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) ||                        \
    defined(USE_ACCELERATE_MPS)
  // Using Apple's Accelerate framework (= BLAS)
  cblas_saxpy((BLASINT)(mat1->rows * mat1->cols), -1.0, mat2->values, 1,
              mat1->values, 1);
#else
#ifdef __ARM_NEON
  size_t n = mat1->rows * mat1->cols;
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t a = vld1q_f32(&mat1->values[i]);
    float32x4_t b = vld1q_f32(&mat2->values[i]);
    vst1q_f32(&mat1->values[i], vsubq_f32(a, b));
  }
  for (; i < n; i++) {
    mat1->values[i] -= mat2->values[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < mat1->rows * mat1->cols; i++) {
    mat1->values[i] -= mat2->values[i];
  }
#endif
#endif
}

void sm_inplace_square_transpose(FloatMatrix *mat) {
  if (mat == NULL || mat->values == NULL || mat->rows != mat->cols) {
    perror("Error: In-place transposition requires a square matrix.");
    return;
  }

  size_t n = mat->rows;

#pragma omp parallel for collapse(2) schedule(dynamic) if (n > 500)
  for (size_t ii = 0; ii < n; ii += BLOCK_SIZE) {
    for (size_t jj = ii; jj < n; jj += BLOCK_SIZE) {
      for (size_t i = ii; i < ii + BLOCK_SIZE && i < n; i++) {
        for (size_t j = (ii == jj ? i + 1 : jj); j < jj + BLOCK_SIZE && j < n;
             j++) {
          float tmp = mat->values[i * n + j];
          mat->values[i * n + j] = mat->values[j * n + i];
          mat->values[j * n + i] = tmp;
        }
      }
    }
  }
}

// In-place scale
void sm_inplace_multiply_by_number(FloatMatrix *mat, const float scalar) {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) ||                        \
    defined(USE_ACCELERATE_MPS)
  cblas_sscal((BLASINT)(mat->rows * mat->cols), scalar, mat->values, 1);
#else
#ifdef __ARM_NEON
  size_t n = mat->rows * mat->cols;
  size_t i = 0;
  float32x4_t s = vdupq_n_f32(scalar);
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(&mat->values[i]);
    vst1q_f32(&mat->values[i], vmulq_f32(v, s));
  }
  for (; i < n; ++i) {
    mat->values[i] *= scalar;
  }
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < mat->rows * mat->cols; i++) {
    mat->values[i] *= scalar;
  }
#endif
#endif
}

// In-place division
void sm_inplace_div(FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !sm_is_equal_size(mat1, mat2)) {
    perror("Error: invalid matrix dimensions for elementwise division.\n");
    return;
  }

  size_t size = mat1->rows * mat1->cols;

  float *a = mat1->values;
  float *b = mat2->values;

#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
  vDSP_vdiv(b, 1, a, 1, a, 1, size);
#else

#ifdef __ARM_NEON
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t va = vld1q_f32(&a[i]);
    float32x4_t vb = vld1q_f32(&b[i]);
    float32x4_t vc = vdivq_f32(va, vb);
    vst1q_f32(&a[i], vc);
  }
  for (; i < size; i++) {
    a[i] /= b[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    a[i] /= b[i];
  }
#endif
#endif
}

FloatMatrix *sm_div(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !sm_is_equal_size(mat1, mat2)) {
    perror("Error: invalid matrix dimensions for elementwise division.\n");
    return NULL;
  }

  FloatMatrix *result = sm_clone(mat1);
  if (!result)
    return NULL;

  sm_inplace_div(result, mat2);
  return result;
}

void sm_inplace_normalize_rows(FloatMatrix *mat) {
  if (!mat || mat->rows == 0 || mat->cols == 0)
    return;

  size_t rows = mat->rows;
  size_t cols = mat->cols;

#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
  for (size_t i = 0; i < rows; ++i) {
    float *row = &mat->values[i * cols];
    float norm;
    vDSP_svesq(row, 1, &norm, cols);
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
      float inv = 1.0f / norm;
      vDSP_vsmul(row, 1, &inv, row, 1, cols);
    }
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < rows; ++i) {
    float norm = 0.0f;
    for (size_t j = 0; j < cols; ++j) {
      float val = mat->values[i * cols + j];
      norm += val * val;
    }
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
      for (size_t j = 0; j < cols; ++j) {
        mat->values[i * cols + j] /= norm;
      }
    }
  }
#endif
}

// Normalize each column of the matrix to unit norm (L2)
void sm_inplace_normalize_cols(FloatMatrix *mat) {
  if (!mat || mat->rows == 0 || mat->cols == 0)
    return;

  size_t rows = mat->rows;
  size_t cols = mat->cols;

#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
  for (size_t j = 0; j < cols; ++j) {
    float norm;
    vDSP_svesq(&mat->values[j], (vDSP_Stride)cols, &norm, (vDSP_Length)rows);
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
      float inv = 1.0f / norm;
      vDSP_vsmul(&mat->values[j], (vDSP_Stride)cols, &inv, &mat->values[j],
                 (vDSP_Stride)cols, (vDSP_Length)rows);
    }
  }
#else
#pragma omp parallel for
  for (size_t j = 0; j < cols; ++j) {
    float norm = 0.0f;
    for (size_t i = 0; i < rows; ++i) {
      float val = mat->values[i * cols + j];
      norm += val * val;
    }
    norm = sqrtf(norm);
    if (norm > 1e-8f) {
      for (size_t i = 0; i < rows; ++i) {
        mat->values[i * cols + j] /= norm;
      }
    }
  }
#endif
}
//
