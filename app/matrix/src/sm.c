/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "sm.h"

#include <log.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <time.h>

#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif

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
static uint64_t sm_global_seed = 0;
static bool sm_seed_initialized = false;

#ifndef M_PI  // on Linux not defined in math.h
#define M_PI 3.14159265358979323846264338327950288f
#endif

/*******************************/
/*      Define Environment     */
/*******************************/

#if defined(USE_ACCELERATE)
#define BLASINT int
#include <Accelerate/Accelerate.h>

#ifdef __APPLE__
#include "sm_mps.h"
#define SM_HAS_MPS 1
#endif

#elif defined(USE_OPENBLAS)
#define BLASINT int
#include <cblas.h>
#include <lapacke.h>
#else
// nothing — OpenMP / ARM NEON fallback
#endif

/* ---- Runtime backend state ---- */
static SmBackend sm_current_backend = SM_BACKEND_DEFAULT;

bool sm_set_backend(SmBackend backend) {
  switch (backend) {
    case SM_BACKEND_DEFAULT:
    case SM_BACKEND_OPENMP:
      sm_current_backend = backend;
      return true;
    case SM_BACKEND_ACCELERATE:
#if defined(USE_ACCELERATE)
      sm_current_backend = backend;
      return true;
#else
      return false;
#endif
    case SM_BACKEND_MPS:
#if defined(SM_HAS_MPS)
      sm_current_backend = backend;
      return true;
#else
      return false;
#endif
    case SM_BACKEND_OPENBLAS:
#if defined(USE_OPENBLAS)
      sm_current_backend = backend;
      return true;
#else
      return false;
#endif
  }
  return false;
}

SmBackend sm_get_backend(void) { return sm_current_backend; }

bool sm_mps_available(void) {
#if defined(SM_HAS_MPS)
  return true;
#else
  return false;
#endif
}

const char *sm_active_library(void) {
  switch (sm_current_backend) {
    case SM_BACKEND_MPS:
#if defined(SM_HAS_MPS)
      return "Metal Performance Shaders";
#endif
      break;
    case SM_BACKEND_ACCELERATE:
#if defined(USE_ACCELERATE)
      return "Apple Accelerate";
#endif
      break;
    case SM_BACKEND_OPENBLAS:
#if defined(USE_OPENBLAS)
      return "OpenBLAS";
#endif
      break;
    case SM_BACKEND_OPENMP:
#if defined(__ARM_NEON)
      return "OpenMP or ARM NEON";
#else
      return "OpenMP";
#endif
    case SM_BACKEND_DEFAULT:
    default:
      break;
  }
  /* DEFAULT: return build-time best */
#if defined(USE_ACCELERATE)
  return "Apple Accelerate";
#elif defined(USE_OPENBLAS)
  return "OpenBLAS";
#elif defined(__ARM_NEON)
  return "OpenMP or ARM NEON";
#else
  return "OpenMP";
#endif
}

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
    log_error("Error: Memory allocation for matrix copy failed.\n");
    return 0;
  }
  memcpy(copy->values, mat->values, rows * cols * sizeof(float));

  /* Inline LU elimination (no pivot matrix, just elimination) */
  FloatMatrix *dummy = copy;
  size_t n = (dummy->rows < dummy->cols) ? dummy->rows : dummy->cols;
  size_t dcols = dummy->cols;
  for (size_t pivot = 0; pivot < n; pivot++) {
    float pivot_val = dummy->values[pivot * dcols + pivot];
    if (fabsf(pivot_val) < EPSILON) continue;
    const float *restrict pivot_row = &dummy->values[pivot * dcols];
#pragma omp parallel for schedule(static) if (n - pivot > 128)
    for (size_t row = pivot + 1; row < n; row++) {
      float factor = dummy->values[row * dcols + pivot] / pivot_val;
      dummy->values[row * dcols + pivot] = 0.0f;
#pragma omp simd
      for (size_t col = pivot + 1; col < dcols; col++) {
        dummy->values[row * dcols + col] -= factor * pivot_row[col];
      }
    }
  }

  size_t rank = 0;
#pragma omp parallel for reduction(+ : rank)
  for (size_t i = 0; i < rows; i++) {
    size_t found = 0;
    for (size_t j = 0; j < cols; j++) {
      if (fabsf(copy->values[i * cols + j]) > EPSILON) {
        found = 1;
      }
    }
    rank += found;
  }

  sm_destroy(copy);
  return rank;
}

static uint64_t sm_mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static uint64_t sm_resolve_seed(uint64_t seed) {
  if (seed != 0) {
    return seed;
  }
  if (sm_seed_initialized) {
    return sm_global_seed;
  }
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (uint64_t)ts.tv_nsec ^ (uint64_t)ts.tv_sec ^
         (uint64_t)(uintptr_t)&sm_global_seed;
}

static float sm_uniform01(uint64_t seed, uint64_t stream, uint64_t idx) {
  uint64_t x = sm_mix64(seed ^ (stream * 0xD2B74407B1CE6E93ull) ^
                        (idx * 0x9E3779B97F4A7C15ull));
  return (float)((double)(x >> 11) / 9007199254740992.0);
}

void sm_set_random_seed(uint64_t seed) {
  sm_global_seed = seed;
  sm_seed_initialized = true;
}

uint64_t sm_get_random_seed(void) {
  if (!sm_seed_initialized) {
    return 0;
  }
  return sm_global_seed;
}

float *sm_to_column_major(const FloatMatrix *mat) {
  if (!mat || !mat->values) {
    log_error("Failed to convert NULL matrix to column-major buffer");
    return NULL;
  }
  size_t rows = mat->rows;
  size_t cols = mat->cols;
  float *col_major = malloc(rows * cols * sizeof(float));
  if (!col_major) {
    log_error("Failed to allocate column-major buffer");
    return NULL;
  }

#pragma omp parallel for collapse(2) if (rows * cols > 250000)
  for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
      col_major[j * rows + i] = mat->values[i * cols + j];
    }
  }
  return col_major;
}

/*******************************/
/*      Public Functions      */
/*******************************/

FloatMatrix *sm_create_empty(void) {
  FloatMatrix *matrix = (FloatMatrix *)malloc(sizeof(FloatMatrix));
  if (!matrix) {
    log_error("Error: could not allocate FloatMatrix struct.\n");
    return NULL;
  }
  matrix->rows = 0;
  matrix->cols = 0;
  matrix->capacity = 0;
  matrix->values = NULL;
  return matrix;
}

FloatMatrix *sm_create_zeros(size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *matrix = (FloatMatrix *)malloc(sizeof(FloatMatrix));
  if (!matrix) {
    log_error("Error: could not allocate FloatMatrix struct.\n");
    return NULL;
  }
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->capacity = rows * cols;
  matrix->values = (float *)calloc(rows * cols, sizeof(float));
  if (!matrix->values) {
    free(matrix);
    log_error("Error: could not allocate values array.\n");
    return NULL;
  }
  return matrix;
}

FloatMatrix *sm_create_with_values(size_t rows, size_t cols, float *values) {
  FloatMatrix *matrix = sm_create(rows, cols);
  if (!matrix) return NULL;
  memcpy(matrix->values, values, rows * cols * sizeof(float));
  return matrix;
}

FloatMatrix *sm_create(size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *matrix = (FloatMatrix *)malloc(sizeof(FloatMatrix));
  if (!matrix) {
    log_error("Error: could not allocate FloatMatrix struct.\n");
    return NULL;
  }
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->capacity = rows * cols;
  matrix->values = (float *)calloc(rows * cols, sizeof(float));
  if (!matrix->values) {
    free(matrix);
    log_error("Error: could not allocate values array.\n");
    return NULL;
  }
  return matrix;
}

FloatMatrix *sm_clone(const FloatMatrix *mat) {
  FloatMatrix *copy = sm_create(mat->rows, mat->cols);
  if (!copy) return NULL;
  memcpy(copy->values, mat->values, mat->rows * mat->cols * sizeof(float));
  return copy;
}

FloatMatrix *sm_create_identity(size_t n) {
  FloatMatrix *identity = sm_create(n, n);
  if (!identity) return NULL;

  for (size_t i = 0; i < n; i++) {
    identity->values[i * n + i] = 1.0f;
  }
  return identity;
}

FloatMatrix *sm_create_random_seeded(size_t rows, size_t cols, uint64_t seed) {
  if (cols != 0 && rows > SIZE_MAX / cols) {
    log_error("Overflow detected in matrix allocation.");
    return NULL;
  }

  FloatMatrix *mat = sm_create(rows, cols);
  if (!mat) {
    return NULL;
  }
  size_t size = rows * cols;
  uint64_t base_seed = sm_resolve_seed(seed);

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    mat->values[i] = sm_uniform01(base_seed, 1, (uint64_t)i);
  }

  return mat;
}

FloatMatrix *sm_create_random(size_t rows, size_t cols) {
  return sm_create_random_seeded(rows, cols, 0);
}

// He initialization (He-et-al.) random matrix creation
FloatMatrix *sm_create_random_he(size_t rows, size_t cols, size_t fan_in) {
  if (cols != 0 && rows > SIZE_MAX / cols) {
    log_error("Overflow detected in matrix allocation.");
    return NULL;
  }
  if (fan_in == 0) {
    log_error("Error: fan_in must be greater than zero.");
    return NULL;
  }

  FloatMatrix *mat = sm_create(rows, cols);
  if (!mat) return NULL;

  float stddev = sqrtf(2.0f / (float)fan_in);
  size_t size = rows * cols;
  uint64_t base_seed = sm_resolve_seed(0);

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    float u1 = sm_uniform01(base_seed, 2, (uint64_t)i);
    if (u1 < 1e-12f) {
      u1 = 1e-12f;
    }
    float u2 = sm_uniform01(base_seed, 3, (uint64_t)i);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    mat->values[i] = z * stddev;
  }

  return mat;
}

// Xavier (Glorot) initialization: Normal distribution
FloatMatrix *sm_create_random_xavier(size_t rows, size_t cols, size_t fan_in,
                                     size_t fan_out) {
  if (cols != 0 && rows > SIZE_MAX / cols) {
    log_error("Overflow detected in matrix allocation.");
    return NULL;
  }
  if (fan_in == 0 || fan_out == 0) {
    log_error("Error: fan_in and fan_out must be greater than zero.");
    return NULL;
  }

  FloatMatrix *mat = sm_create(rows, cols);
  if (!mat) return NULL;

  float stddev = sqrtf(2.0f / (float)(fan_in + fan_out));
  size_t size = rows * cols;
  uint64_t base_seed = sm_resolve_seed(0);

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    float u1 = sm_uniform01(base_seed, 4, (uint64_t)i);
    if (u1 < 1e-12f) {
      u1 = 1e-12f;
    }
    float u2 = sm_uniform01(base_seed, 5, (uint64_t)i);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    mat->values[i] = z * stddev;
  }

  return mat;
}

FloatMatrix *sm_from_array_ptrs(size_t rows, size_t cols, float **array) {
  // check if the array is NULL
  if (array == NULL) {
    log_error("Error: array is NULL.\n");
    return NULL;
  }

  FloatMatrix *mat = sm_create(rows, cols);
  if (!mat) return NULL;

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
    log_error("Error: array is NULL.\n");
    return NULL;
  }

  FloatMatrix *matrix = sm_create(rows, cols);
  if (!matrix) return NULL;

#pragma omp parallel for
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      matrix->values[i * cols + j] = array[i][j];
    }
  }
  return matrix;
}

float *sm_to_array(FloatMatrix *matrix) {
  if (matrix == NULL || matrix->values == NULL) {
    log_error("Error: matrix is NULL.\n");
    return NULL;
  }

  float *array = (float *)malloc(matrix->rows * matrix->cols * sizeof(float));
  if (!array) {
    log_error("Error: could not allocate array.\n");
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

float *sm_create_array_from_matrix(FloatMatrix *matrix) {
  return sm_to_array(matrix);
}

FloatMatrix *sm_get_row(const FloatMatrix *mat, size_t i) {
  FloatMatrix *row = sm_create(1, mat->cols);
  if (!row) return NULL;
  memcpy(row->values, &mat->values[i * mat->cols], mat->cols * sizeof(float));
  return row;
}

FloatMatrix *sm_get_last_row(const FloatMatrix *mat) {
  return sm_get_row(mat, mat->rows - 1);
}

FloatMatrix *sm_get_col(const FloatMatrix *mat, size_t j) {
  FloatMatrix *col = sm_create(mat->rows, 1);
  if (!col) return NULL;
  for (size_t i = 0; i < mat->rows; i++) {
    col->values[i] = mat->values[i * mat->cols + j];
  }
  return col;
}

FloatMatrix *sm_get_last_col(const FloatMatrix *mat) {
  return sm_get_col(mat, mat->cols - 1);
}

// Performance & usage guidance: see .github/skills/sm-gemm-optimization/SKILL.md
bool sm_gemm(FloatMatrix *C, float alpha, const FloatMatrix *A,
             SmTranspose trans_a, const FloatMatrix *B, SmTranspose trans_b,
             float beta) {
  if (!A || !B || !C || !A->values || !B->values || !C->values) {
    log_error("Error: sm_gemm received NULL input.");
    return false;
  }

  size_t m = (trans_a == SM_TRANSPOSE) ? A->cols : A->rows;
  size_t k_a = (trans_a == SM_TRANSPOSE) ? A->rows : A->cols;
  size_t k_b = (trans_b == SM_TRANSPOSE) ? B->cols : B->rows;
  size_t n = (trans_b == SM_TRANSPOSE) ? B->rows : B->cols;

  if (k_a != k_b) {
    log_error("Error: sm_gemm inner dimensions mismatch.");
    return false;
  }
  if (C->rows != m || C->cols != n) {
    log_error("Error: sm_gemm output dimensions mismatch.");
    return false;
  }

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)

#if defined(SM_HAS_MPS)
  /* Runtime MPS dispatch: only when backend explicitly set to MPS */
  if (sm_current_backend == SM_BACKEND_MPS) {
    if (mps_matrix_multiply_ex(A->values, A->rows, A->cols,
                               trans_a == SM_TRANSPOSE, B->values, B->rows,
                               B->cols, trans_b == SM_TRANSPOSE, alpha, beta,
                               C->values, C->rows, C->cols)) {
      return true;
    }
    /* MPS failed — fall through to cblas */
  }
#endif

  enum CBLAS_TRANSPOSE op_a =
      (trans_a == SM_TRANSPOSE) ? CblasTrans : CblasNoTrans;
  enum CBLAS_TRANSPOSE op_b =
      (trans_b == SM_TRANSPOSE) ? CblasTrans : CblasNoTrans;

  cblas_sgemm(CblasRowMajor, op_a, op_b, (BLASINT)m, (BLASINT)n, (BLASINT)k_a,
              alpha, A->values, (BLASINT)A->cols, B->values, (BLASINT)B->cols,
              beta, C->values, (BLASINT)C->cols);
#else
  size_t k = k_a;
  float *restrict c = C->values;
  const float *restrict a = A->values;
  const float *restrict b = B->values;

  /* --- Initialise C = beta * C --- */
  if (beta == 0.0f) {
    memset(c, 0, m * n * sizeof(float));
  } else if (beta != 1.0f) {
    for (size_t i = 0; i < m * n; i++) {
      c[i] *= beta;
    }
  }

  /* --- Fast path: NoTrans x NoTrans (i-k-j order with tiling) --- */
  if (trans_a == SM_NO_TRANSPOSE && trans_b == SM_NO_TRANSPOSE) {
    enum { TILE = 64 };

#pragma omp parallel for schedule(static)
    for (size_t ii = 0; ii < m; ii += TILE) {
      const size_t i_end = (ii + TILE < m) ? ii + TILE : m;
      for (size_t kk = 0; kk < k; kk += TILE) {
        const size_t k_end = (kk + TILE < k) ? kk + TILE : k;
        for (size_t jj = 0; jj < n; jj += TILE) {
          const size_t j_end = (jj + TILE < n) ? jj + TILE : n;
          for (size_t i = ii; i < i_end; i++) {
            for (size_t p = kk; p < k_end; p++) {
              const float a_ip = alpha * a[i * A->cols + p];
#pragma omp simd
              for (size_t j = jj; j < j_end; j++) {
                c[i * n + j] += a_ip * b[p * B->cols + j];
              }
            }
          }
        }
      }
    }
  } else {
    /* --- General path for transpose cases --- */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; i++) {
      for (size_t p = 0; p < k; p++) {
        const float a_ip =
            alpha * ((trans_a == SM_TRANSPOSE) ? a[p * A->cols + i]
                                               : a[i * A->cols + p]);
#pragma omp simd
        for (size_t j = 0; j < n; j++) {
          const float b_pj = (trans_b == SM_TRANSPOSE)
                                  ? b[j * B->cols + p]
                                  : b[p * B->cols + j];
          c[i * n + j] += a_ip * b_pj;
        }
      }
    }
  }
#endif

  return true;
}

bool sm_gemm_bias_relu(FloatMatrix *C, const FloatMatrix *A, SmTranspose trans_a,
                       const FloatMatrix *B, SmTranspose trans_b,
                       const FloatMatrix *bias) {
  if (!sm_gemm(C, 1.0f, A, trans_a, B, trans_b, 0.0f)) {
    return false;
  }

  if (!C || !C->values) {
    log_error("Error: sm_gemm_bias_relu received NULL output.");
    return false;
  }

  const size_t rows = C->rows;
  const size_t cols = C->cols;
  float *c = C->values;

  if (bias) {
    if (!bias->values || bias->cols != cols ||
        !(bias->rows == 1 || bias->rows == rows)) {
      log_error("Error: sm_gemm_bias_relu bias shape mismatch.");
      return false;
    }

    const float *b = bias->values;
    if (bias->rows == 1) {
      for (size_t i = 0; i < rows; ++i) {
        size_t row_offset = i * cols;
        for (size_t j = 0; j < cols; ++j) {
          float v = c[row_offset + j] + b[j];
          c[row_offset + j] = (v > 0.0f) ? v : 0.0f;
        }
      }
    } else {
      for (size_t i = 0; i < rows; ++i) {
        size_t row_offset = i * cols;
        size_t bias_offset = i * cols;
        for (size_t j = 0; j < cols; ++j) {
          float v = c[row_offset + j] + b[bias_offset + j];
          c[row_offset + j] = (v > 0.0f) ? v : 0.0f;
        }
      }
    }
  } else {
    for (size_t i = 0; i < rows * cols; ++i) {
      if (c[i] < 0.0f) c[i] = 0.0f;
    }
  }

  return true;
}

FloatMatrix *sm_multiply(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || mat1->cols != mat2->rows) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }

  FloatMatrix *product = sm_create(mat1->rows, mat2->cols);
  if (!product) {
    return NULL;
  }

  if (!sm_gemm(product, 1.0f, mat1, SM_NO_TRANSPOSE, mat2, SM_NO_TRANSPOSE,
               0.0f)) {
    sm_destroy(product);
    return NULL;
  }

  return product;
}

bool sm_inplace_elementwise_multiply(FloatMatrix *mat1,
                                     const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !sm_is_equal_size(mat1, mat2)) {
    log_error("Error: invalid matrix dimensions for Hadamard product.\n");
    return false;
  }

  size_t size = mat1->rows * mat1->cols;

#if defined(USE_ACCELERATE)

  vDSP_vmul(mat1->values, 1, mat2->values, 1, mat1->values, 1, size);
  return true;
#else

  float *a = mat1->values;
  const float *b = mat2->values;

#pragma omp parallel for schedule(static)
  for (size_t blk = 0; blk < size; blk += 256) {
    size_t end = (blk + 256 < size) ? blk + 256 : size;
#ifdef __ARM_NEON
    size_t i = blk;
    for (; i + 4 <= end; i += 4) {
      vst1q_f32(&a[i], vmulq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < end; i++) { a[i] *= b[i]; }
#else
#pragma omp simd
    for (size_t i = blk; i < end; i++) { a[i] *= b[i]; }
#endif
  }
#endif

  return true;
}

FloatMatrix *sm_elementwise_multiply(const FloatMatrix *mat1,
                                     const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !sm_is_equal_size(mat1, mat2)) {
    log_error("Error: invalid matrix dimensions for Hadamard product.\n");
    return NULL;
  }

  FloatMatrix *result = sm_clone(mat1);
  if (!result) return NULL;

  if (!sm_inplace_elementwise_multiply(result, mat2)) {
    sm_destroy(result);
    return NULL;
  }
  return result;
}

FloatMatrix *sm_multiply_by_number(const FloatMatrix *mat, const float number) {
  FloatMatrix *product = sm_clone(mat);
  if (!product) return NULL;
  if (!sm_inplace_multiply_by_number(product, number)) {
    sm_destroy(product);
    return NULL;
  }
  return product;
}

FloatMatrix *sm_transpose(const FloatMatrix *mat) {
  if (mat == NULL || mat->values == NULL) return NULL;

  if (sm_is_square(mat)) {
    FloatMatrix *copy = sm_clone(mat);
    if (!copy || !sm_inplace_square_transpose(copy)) {
      sm_destroy(copy);
      return NULL;
    }
    return copy;
  }

  size_t n = mat->rows;
  size_t m = mat->cols;

  FloatMatrix *transposed = sm_create(m, n);
  if (!transposed) return NULL;

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
    log_error("Error: invalid matrix dimensions for solve.\n");
    return NULL;
  }
#if defined(USE_ACCELERATE)
  int n = (int)A->rows;
  int nrhs = (int)b->cols;
  int lda = n;
  int ldb = n;

  FloatMatrix *a_copy = sm_clone(A);
  FloatMatrix *b_copy = sm_clone(b);
  if (!a_copy || !b_copy) {
    log_error("Error: Memory allocation for solve-system copies failed.\n");
    sm_destroy(a_copy);
    sm_destroy(b_copy);
    return NULL;
  }

  float *a_col = sm_to_column_major(a_copy);
  float *b_col = sm_to_column_major(b_copy);
  if (!a_col || !b_col) {
    log_error(
        "Error: Memory allocation for solve-system column-major buffers failed.\n");
    free(a_col);
    free(b_col);
    sm_destroy(a_copy);
    sm_destroy(b_copy);
    return NULL;
  }

  int *ipiv = (int *)malloc((size_t)n * sizeof(int));
  if (!ipiv) {
    log_error("Error: Memory allocation for pivot indices failed.\n");
    free(a_col);
    free(b_col);
    sm_destroy(a_copy);
    sm_destroy(b_copy);
    return NULL;
  }
  int info;

  // Solve AX = B using LAPACK
  sgesv_(&n, &nrhs, a_col, &lda, ipiv, b_col, &ldb, &info);

  free(ipiv);
  free(a_col);
  sm_destroy(a_copy);

  if (info != 0) {
    log_error("Error: sgesv_ failed with info = %d\n", info);
    sm_destroy(b_copy);
    free(b_col);
    return NULL;
  }

  float *restrict dst = b_copy->values;
  const float *restrict src = b_col;
  size_t rows = b_copy->rows;
  size_t cols = b_copy->cols;

  /* Column-major → row-major transpose.  No OMP here: the matrix is
     typically small (nrhs columns) and Accelerate manages its own
     thread pool — spawning OMP threads would cause oversubscription. */
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
  if (!a_copy || !b_copy) {
    log_error("Error: Memory allocation for solve-system copies failed.\n");
    sm_destroy(a_copy);
    sm_destroy(b_copy);
    return NULL;
  }

  int *ipiv = (int *)malloc((size_t)n * sizeof(int));
  if (!ipiv) {
    log_error("Error: Memory allocation for pivot indices failed.\n");
    sm_destroy(a_copy);
    sm_destroy(b_copy);
    return NULL;
  }
  int info;

  info = LAPACKE_sgesv(LAPACK_ROW_MAJOR,
                       n,     // number of equations
                       nrhs,  // number of right-hand sides
                       a_copy->values,
                       lda,  // leading dimension of A (= cols in row-major)
                       ipiv, b_copy->values,
                       ldb  // leading dimension of B (= cols in row-major)
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
    log_error("Error: LU decomposition failed.\n");
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

  /* Transpose RHS so each column becomes a contiguous row
     to avoid false sharing when parallelising over columns. */
  float *restrict rt = (float *)malloc(rhs * n * sizeof(float));
  if (!rt) {
    sm_destroy(lu);
    sm_destroy(x);
    free(pivot_order);
    return NULL;
  }
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < rhs; ++k) {
      rt[k * n + i] = x->values[i * rhs + k];
    }
  }

  const float *restrict lv = lu->values;

#pragma omp parallel for schedule(static) if (rhs > 1 && n > 128)
  for (size_t k = 0; k < rhs; ++k) {
    float *restrict r = &rt[k * n];

    /* Forward substitution (L * y = Pb) */
    for (size_t i = 1; i < n; ++i) {
      float sum = r[i];
      for (size_t j = 0; j < i; ++j) {
        sum -= lv[i * n + j] * r[j];
      }
      r[i] = sum;
    }

    /* Back substitution (U * x = y) */
    size_t i = n;
    do {
      --i;
      float sum = r[i];
      for (size_t j = i + 1; j < n; ++j) {
        sum -= lv[i * n + j] * r[j];
      }
      r[i] = sum / lv[i * n + i];
    } while (i != 0);
  }

  /* Transpose result back into x */
  for (size_t k = 0; k < rhs; ++k) {
    const float *restrict r = &rt[k * n];
    for (size_t i = 0; i < n; ++i) {
      x->values[i * rhs + k] = r[i];
    }
  }
  free(rt);

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
  if (!mat1 || !mat2 || mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *sum = sm_clone(mat1);
  if (!sum || !sm_inplace_add(sum, mat2)) {
    sm_destroy(sum);
    return NULL;
  }
  return sum;
}

FloatMatrix *sm_diff(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *difference = sm_clone(mat1);
  if (!difference || !sm_inplace_diff(difference, mat2)) {
    sm_destroy(difference);
    return NULL;
  }
  return difference;
}

// Use TOLERANCE for numerical zero threshold in LU decomposition
#define TOLERANCE EPSILON
bool sm_lu_decompose(FloatMatrix *mat, size_t *pivot_order) {
  size_t n = mat->rows;
  if (mat->cols != n) return false;

  for (size_t pivot = 0; pivot < n; pivot++) {
    float max_val = (float)fabsf(mat->values[pivot * n + pivot]);
    size_t max_row = pivot;
    for (size_t row = pivot + 1; row < n; row++) {
      float val = (float)fabsf(mat->values[row * n + pivot]);
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
    log_error("the Matrix has to be square!");
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
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)

    BLASINT *ipiv = (BLASINT *)malloc(mat->cols * sizeof(BLASINT));
    FloatMatrix *lu = sm_clone(mat);
    if (!ipiv || !lu) {
      log_error("Error: Memory allocation for determinant failed.\n");
      free(ipiv);
      sm_destroy(lu);
      return 0.0f;
    }
    BLASINT info = 0;
    BLASINT cols = (BLASINT)lu->cols;
    BLASINT rows = (BLASINT)lu->rows;

    sgetrf_(&cols, &rows, lu->values, &cols, ipiv, &info);
    if (info != 0) {
      log_error("Error: dgetrf failed.\n");
      free(ipiv);
      sm_destroy(lu);
      return 0;
    }
    float det = 1.0f;
    for (size_t i = 0; i < mat->cols; i++) {
      det *= sm_get(lu, i, i);
      if (ipiv[i] != (BLASINT)(i + 1)) det = -det;
    }
    free(ipiv);
    sm_destroy(lu);

#else
    FloatMatrix *copy = sm_clone(mat);
    if (!copy) return 0.0f;

    size_t n = mat->rows;
    float *a = copy->values;
    int sign = 1;

    for (size_t k = 0; k < n; ++k) {
      /* Partial pivoting */
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
        for (size_t j = k; j < n; ++j) {
          float tmp = a[k * n + j];
          a[k * n + j] = a[max_row * n + j];
          a[max_row * n + j] = tmp;
        }
        sign = -sign;
      }

      const float pivot = a[k * n + k];
      const float *restrict pivot_row = &a[k * n];
#pragma omp parallel for schedule(static) if (n - k > 128)
      for (size_t i = k + 1; i < n; ++i) {
        float factor = a[i * n + k] / pivot;
        a[i * n + k] = 0.0f;
#pragma omp simd
        for (size_t j = k + 1; j < n; ++j) {
          a[i * n + j] -= factor * pivot_row[j];
        }
      }
    }

    float det = (float)sign;
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
    log_error("the Matrix has to be square!");
    return NULL;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  FloatMatrix *inverse = sm_clone(mat);
  if (inverse == NULL) {
    log_error("Error: Memory allocation for inverse copy failed.\n");
    return NULL;
  }
  BLASINT *ipiv = (BLASINT *)malloc(mat->cols * sizeof(BLASINT));
  if (ipiv == NULL) {
    sm_destroy(inverse);
    log_error("Error: Memory allocation for ipiv failed.\n");
    return NULL;
  }
  BLASINT info = 0;
  BLASINT n = (BLASINT)inverse->cols;

  sgetrf_(&n, &n, inverse->values, &n, ipiv, &info);
  if (info != 0) {
    free(ipiv);
    sm_destroy(inverse);
    log_error("Error: dgetrf failed.\n");
    return NULL;
  }

  BLASINT lwork = -1;
  float work_opt;
  sgetri_(&n, inverse->values, &n, ipiv, &work_opt, &lwork, &info);
  if (info != 0 || work_opt <= 0.0f) {
    free(ipiv);
    sm_destroy(inverse);
    log_error("Error: dgetri workspace query failed.\n");
    return NULL;
  }

  lwork = (BLASINT)work_opt;
  float *work = (float *)malloc((size_t)lwork * sizeof(float));
  if (work == NULL) {
    free(ipiv);
    sm_destroy(inverse);
    log_error("Error: Memory allocation for work array failed.\n");
    return NULL;
  }

  sgetri_(&n, inverse->values, &n, ipiv, work, &lwork, &info);
  free(work);
  free(ipiv);
  if (info != 0) {
    sm_destroy(inverse);
    log_error("Error: dgetri failed.\n");
    return NULL;
  }
  return inverse;
#else
  // Neue Nicht-BLAS-Variante: LU-Zerlegung und
  // Vorwärts-/Rückwärtssubstitution
  if (!sm_is_square(mat)) {
    log_error("Error: Matrix must be square.\n");
    return NULL;
  }

  size_t n = mat->cols;
  FloatMatrix *copy = sm_clone(mat);
  if (!copy) return NULL;

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
    log_error("Error: LU decomposition failed.\n");
    free(pivot_order);
    sm_destroy(copy);
    sm_destroy(inverse);
    return NULL;
  }

  // Apply pivot permutation to the identity matrix (right-hand side)
  for (size_t i = 0; i < n; ++i) {
    if (pivot_order[i] != i) {
      for (size_t j = 0; j < n; ++j) {
        float tmp = inverse->values[i * n + j];
        inverse->values[i * n + j] = inverse->values[pivot_order[i] * n + j];
        inverse->values[pivot_order[i] * n + j] = tmp;
      }
    }
  }

  /* Transpose RHS so each "column" is a contiguous row → avoids false sharing */
  float *restrict rhs = (float *)malloc(n * n * sizeof(float));
  if (!rhs) {
    free(pivot_order);
    sm_destroy(copy);
    sm_destroy(inverse);
    return NULL;
  }
  const float *restrict inv = inverse->values;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      rhs[j * n + i] = inv[i * n + j];  /* transpose: rhs[col][row] */
    }
  }

  const float *restrict lu = copy->values;

#pragma omp parallel for schedule(static) if (n > 128)
  for (size_t col = 0; col < n; col++) {
    float *restrict r = &rhs[col * n]; /* this "column" is now a contiguous row */

    /* Forward substitution (L * y = e_col) */
    for (size_t i = 1; i < n; i++) {
      float sum = r[i];
      for (size_t j = 0; j < i; j++) {
        sum -= lu[i * n + j] * r[j];
      }
      r[i] = sum;
    }

    /* Back substitution (U * x = y) */
    size_t i = n;
    do {
      --i;
      float sum = r[i];
      for (size_t j = i + 1; j < n; j++) {
        sum -= lu[i * n + j] * r[j];
      }
      r[i] = sum / lu[i * n + i];
    } while (i != 0);
  }

  /* Transpose result back */
  float *restrict dst = inverse->values;
  for (size_t col = 0; col < n; ++col) {
    const float *restrict r = &rhs[col * n];
    for (size_t i = 0; i < n; ++i) {
      dst[i * n + col] = r[i];
    }
  }
  free(rhs);

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
    log_error("Error: could not reallocate memory for dense matrix.\n");
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
  if (!slice) return NULL;

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
  printf("Matrix[Float.2f] (%zu x %zu)\n", matrix->rows, matrix->cols);
}

void sm_destroy(FloatMatrix *mat) {
  if (!mat) return;
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
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
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
    return 0;  // No matrix, no rank
  }
  size_t rank = 0;
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  // Work on a copy to avoid modifying the const input
  FloatMatrix *qr_copy = sm_clone(mat);
  if (!qr_copy) return 0;

  BLASINT m = (BLASINT)qr_copy->rows;
  BLASINT n = (BLASINT)qr_copy->cols;
  BLASINT lda = n;
  BLASINT lwork = -1;
  float wkopt;
  float *work;
  BLASINT info;

  sgeqrf_(&m, &n, qr_copy->values, &lda, NULL, &wkopt, &lwork, &info);
  lwork = (BLASINT)wkopt;
  work = (float *)malloc((size_t)lwork * sizeof(float));
  if (work == NULL) {
    sm_destroy(qr_copy);
    return 0;  // Memory allocation failed
  }

  float *tau = (float *)malloc((size_t)(m < n ? m : n) * sizeof(float));
  if (tau == NULL) {
    free(work);
    sm_destroy(qr_copy);
    return 0;  // Memory allocation failed
  }

  sgeqrf_(&m, &n, qr_copy->values, &lda, tau, work, &lwork, &info);
  free(work);
  free(tau);
  if (info != 0) {
    sm_destroy(qr_copy);
    return 0;  // QR factorization failed
  }

  int k = (m < n) ? m : n;
  for (int i = 0; i < k; ++i) {
    if (fabsf(qr_copy->values[i * lda + i]) > EPSILON) {
      rank++;
    }
  }
  sm_destroy(qr_copy);
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
    // vcgtq_f32 yields 0xFFFFFFFF per true lane; reinterpret as -1 (int32)
    counter -= (int)vaddvq_s32(vreinterpretq_s32_u32(cmp));
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
bool sm_inplace_add(FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !mat1->values || !mat2->values) {
    log_error("Error: invalid matrix input.\n");
    return false;
  }

  if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
    log_error("Error: invalid matrix dimensions.\n");
    return false;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_saxpy((BLASINT)(mat1->rows * mat1->cols), 1.0, mat2->values, 1,
              mat1->values, 1);
#else
  size_t total = mat1->rows * mat1->cols;
  float *restrict a = mat1->values;
  const float *restrict b = mat2->values;
#pragma omp parallel for schedule(static)
  for (size_t blk = 0; blk < total; blk += 256) {
    size_t end = (blk + 256 < total) ? blk + 256 : total;
#ifdef __ARM_NEON
    size_t i = blk;
    for (; i + 4 <= end; i += 4) {
      vst1q_f32(&a[i], vaddq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < end; i++) { a[i] += b[i]; }
#else
#pragma omp simd
    for (size_t i = blk; i < end; i++) { a[i] += b[i]; }
#endif
  }
#endif

  return true;
}

// In-place difference
bool sm_inplace_diff(FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !mat1->values || !mat2->values) {
    log_error("Error: invalid matrix input.\n");
    return false;
  }

  if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
    log_error("Error: invalid matrix dimensions.\n");
    return false;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  // Using Apple's Accelerate framework (= BLAS)
  cblas_saxpy((BLASINT)(mat1->rows * mat1->cols), -1.0, mat2->values, 1,
              mat1->values, 1);
#else
  size_t total = mat1->rows * mat1->cols;
  float *restrict a = mat1->values;
  const float *restrict b = mat2->values;
#pragma omp parallel for schedule(static)
  for (size_t blk = 0; blk < total; blk += 256) {
    size_t end = (blk + 256 < total) ? blk + 256 : total;
#ifdef __ARM_NEON
    size_t i = blk;
    for (; i + 4 <= end; i += 4) {
      vst1q_f32(&a[i], vsubq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < end; i++) { a[i] -= b[i]; }
#else
#pragma omp simd
    for (size_t i = blk; i < end; i++) { a[i] -= b[i]; }
#endif
  }
#endif

  return true;
}

bool sm_inplace_square_transpose(FloatMatrix *mat) {
  if (mat == NULL || mat->values == NULL || mat->rows != mat->cols) {
    log_error("Error: In-place transposition requires a square matrix.");
    return false;
  }

  size_t n = mat->rows;

#pragma omp parallel for collapse(2) if (n > 500)
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

  return true;
}

// In-place scale
bool sm_inplace_multiply_by_number(FloatMatrix *mat, const float scalar) {
  if (!mat || !mat->values) {
    log_error("Error: invalid matrix input.\n");
    return false;
  }

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_sscal((BLASINT)(mat->rows * mat->cols), scalar, mat->values, 1);
#else
  size_t total = mat->rows * mat->cols;
  float *restrict a = mat->values;
#pragma omp parallel for schedule(static)
  for (size_t blk = 0; blk < total; blk += 256) {
    size_t end = (blk + 256 < total) ? blk + 256 : total;
#ifdef __ARM_NEON
    float32x4_t s = vdupq_n_f32(scalar);
    size_t i = blk;
    for (; i + 4 <= end; i += 4) {
      vst1q_f32(&a[i], vmulq_f32(vld1q_f32(&a[i]), s));
    }
    for (; i < end; ++i) { a[i] *= scalar; }
#else
#pragma omp simd
    for (size_t i = blk; i < end; i++) { a[i] *= scalar; }
#endif
  }
#endif

  return true;
}

// In-place division
bool sm_inplace_div(FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !sm_is_equal_size(mat1, mat2)) {
    log_error("Error: invalid matrix dimensions for elementwise division.\n");
    return false;
  }

  size_t size = mat1->rows * mat1->cols;

  float *a = mat1->values;
  float *b = mat2->values;

#if defined(USE_ACCELERATE)
  vDSP_vdiv(b, 1, a, 1, a, 1, size);
#else

#pragma omp parallel for schedule(static)
  for (size_t blk = 0; blk < size; blk += 256) {
    size_t end = (blk + 256 < size) ? blk + 256 : size;
#ifdef __ARM_NEON
    size_t i = blk;
    for (; i + 4 <= end; i += 4) {
      vst1q_f32(&a[i], vdivq_f32(vld1q_f32(&a[i]), vld1q_f32(&b[i])));
    }
    for (; i < end; i++) { a[i] /= b[i]; }
#else
#pragma omp simd
    for (size_t i = blk; i < end; i++) { a[i] /= b[i]; }
#endif
  }
#endif

  return true;
}

FloatMatrix *sm_div(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (!mat1 || !mat2 || !sm_is_equal_size(mat1, mat2)) {
    log_error("Error: invalid matrix dimensions for elementwise division.\n");
    return NULL;
  }

  FloatMatrix *result = sm_clone(mat1);
  if (!result) return NULL;

  if (!sm_inplace_div(result, mat2)) {
    sm_destroy(result);
    return NULL;
  }
  return result;
}

bool sm_inplace_normalize_rows(FloatMatrix *mat) {
  if (!mat || !mat->values || mat->rows == 0 || mat->cols == 0) {
    return false;
  }

  size_t rows = mat->rows;
  size_t cols = mat->cols;

#if defined(USE_ACCELERATE)
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

  return true;
}

// Normalize each column of the matrix to unit norm (L2)
bool sm_inplace_normalize_cols(FloatMatrix *mat) {
  if (!mat || !mat->values || mat->rows == 0 || mat->cols == 0) {
    return false;
  }

  size_t rows = mat->rows;
  size_t cols = mat->cols;

#if defined(USE_ACCELERATE)
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

  return true;
}
//
