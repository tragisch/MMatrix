/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "sm.h"
#include <omp.h>
#include <stdint.h>
#include <time.h>

#define INIT_CAPACITY 100
#define EPSILON 1e-5

/*******************************/
/*      Define Environment     */
/*******************************/

#ifdef __APPLE__
#if !defined(USE_ACCELERATE) && !defined(USE_OPENBLAS) && !defined(USE_ACCELERATE_MPS)
#define USE_ACCELERATE_MPS 1
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
#else
#include <arm_neon.h>
#include <math.h>

#endif

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
    for (size_t j = 0; j < cols; j++) {
      if (fabs(copy->values[i * cols + j]) > EPSILON) {
        rank++;
        break;
      }
    }
  }

  sm_destroy(copy);
  return rank;
}

static unsigned int sm_random_seed() {
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

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      col_major[j * rows + i] = mat->values[i * cols + j];
    }
  }
  return col_major;
}

/*******************************/
/*      Public Functions      */
/*******************************/

char *sm_active_library() {
#ifdef USE_ACCELERATE
  return "Apple's Accelerate";
#elif defined(USE_ACCELERATE_MPS)
  return "Apple's Accelerate MPS";
#elif defined(USE_OPENBLAS)
  return "OpenBLAS";
#else
  return "No BLAS";
#endif
}

FloatMatrix *sm_create_empty() {
  FloatMatrix *matrix = (FloatMatrix *)malloc(sizeof(FloatMatrix));
  matrix->rows = 0;
  matrix->cols = 0;
  matrix->capacity = 0;
  matrix->values = NULL;
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

FloatMatrix *sm_create_clone(const FloatMatrix *mat) {
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

  unsigned int global_seed = sm_random_seed() ^ (uintptr_t)mat;
#pragma omp parallel
  {
    unsigned int seed = global_seed ^ omp_get_thread_num();
#pragma omp for
    for (size_t i = 0; i < size; ++i) {
      mat->values[i] = (float)rand_r(&seed) / RAND_MAX;
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

  unsigned int global_seed = sm_random_seed() ^ (uintptr_t)mat;
#pragma omp parallel
  {
    unsigned int seed = global_seed ^ omp_get_thread_num();
#pragma omp for
    for (size_t i = 0; i < size; ++i) {
      // Box-Muller Transform (approximate standard normal)
      float u1 = (float)rand_r(&seed) / RAND_MAX;
      float u2 = (float)rand_r(&seed) / RAND_MAX;
      float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
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

  unsigned int global_seed = sm_random_seed() ^ (uintptr_t)mat;
#pragma omp parallel
  {
    unsigned int seed = global_seed ^ omp_get_thread_num();
#pragma omp for
    for (size_t i = 0; i < size; ++i) {
      // Box-Muller Transform: generate standard normal distributed value
      float u1 = (float)rand_r(&seed) / RAND_MAX;
      float u2 = (float)rand_r(&seed) / RAND_MAX;
      float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
      mat->values[i] = z * stddev;
    }
  }

  return mat;
}

FloatMatrix *sm_create_from_array(size_t rows, size_t cols, float **array) {
  // check if the array is NULL
  if (array == NULL) {
    perror("Error: array is NULL.\n");
    return NULL;
  }
  // check if array is an array of arrays
  if (array[0] == NULL) {
    perror("Error: array is not an array of arrays.\n");
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

FloatMatrix *sm_create_from_2D_array(size_t rows, size_t cols,
                                     float array[rows][cols]) {
  // check if the array is NULL
  if (array == NULL) {
    perror("Error: array is NULL.\n");
    return NULL;
  }
  // check if array is an 2D array
  if (array[0] == NULL) {
    perror("Error: array is not an 2D array.\n");
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

double *sm_create_array_from_matrix(FloatMatrix *matrix) {
  if (matrix == NULL || matrix->values == NULL) {
    perror("Error: matrix is NULL.\n");
    return NULL;
  }

  double *array =
      (double *)malloc(matrix->rows * matrix->cols * sizeof(double));
  if (!array) {
    perror("Error: could not allocate array.\n");
    return NULL;
  }

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < matrix->rows; ++i) {
    for (size_t j = 0; j < matrix->cols; ++j) {
      array[i * matrix->cols + j] =
          (double)matrix->values[i * matrix->cols + j];
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
#ifdef USE_ACCELERATE

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

#elif defined(USE_OPENBLAS)

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (BLASINT)mat1->rows,
              (BLASINT)mat2->cols, (BLASINT)mat1->cols, 1.0, mat1->values,
              (BLASINT)mat1->cols, mat2->values, (BLASINT)mat2->cols, 0.0,
              product->values, (BLASINT)product->cols);

#else
  // ohne SMID
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

FloatMatrix *sm_multiply_by_number(const FloatMatrix *mat, const float number) {
  FloatMatrix *product = sm_create_clone(mat);
  sm_inplace_multiply_by_number(product, number);
  return product;
}

FloatMatrix *sm_transpose(const FloatMatrix *mat) {
  if (mat == NULL || mat->values == NULL)
    return NULL;

  if (sm_is_square(mat)) {
    FloatMatrix *copy = sm_create_clone(mat);
    sm_inplace_square_transpose(copy);
    return copy;
  }

  FloatMatrix *transposed = sm_create(mat->cols, mat->rows);

  float *src = mat->values;
  float *dst = transposed->values;

#pragma omp parallel for collapse(2) schedule(dynamic)
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

bool sm_is_equal(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (mat1 == NULL || mat2 == NULL) {
    return false;
  }
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    return false;
  }

  size_t size = mat1->cols * mat1->rows;
  int equal = 1;

#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    if (fabs(mat1->values[i] - mat2->values[i]) > EPSILON) {
#pragma omp atomic write
      equal = 0;
    }
  }

  return equal;
}

FloatMatrix *sm_add(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *sum = sm_create_clone(mat1);
  sm_inplace_add(sum, mat2);

  return sum;
}

FloatMatrix *sm_diff(const FloatMatrix *mat1, const FloatMatrix *mat2) {
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  FloatMatrix *difference = sm_create_clone(mat1);
  sm_inplace_diff(difference, mat2);
  return difference;
}

#if !defined(USE_ACCELERATE) && !defined(USE_OPENBLAS) &&                      \
    !defined(USE_ACCELERATE_MPS)
static bool sm_lu_decompose(FloatMatrix *mat, size_t *pivot_order) {
  size_t n = mat->rows;
  if (mat->cols != n)
    return false;

  for (size_t pivot = 0; pivot < n; pivot++) {
    float max_val = fabs(mat->values[pivot * n + pivot]);
    size_t max_row = pivot;
    for (size_t row = pivot + 1; row < n; row++) {
      float val = fabs(mat->values[row * n + pivot]);
      if (val > max_val) {
        max_val = val;
        max_row = row;
      }
    }

    if (max_val < EPSILON) {
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
#endif

float sm_determinant(const FloatMatrix *mat) {
  if (mat->cols != mat->rows) {
    perror("the Matrix has to be square!");
  }
  float det = 0;
  if (mat->cols == 1) {
    return sm_get(mat, 0, 0);
  } else if (mat->cols == 2) {
    return sm_get(mat, 0, 0) * sm_get(mat, 1, 1) -
           sm_get(mat, 0, 1) * sm_get(mat, 1, 0);
  } else if (mat->cols == 3) {
    float *a = mat->values;
    det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
          a[1] * (a[3] * a[8] - a[5] * a[6]) +
          a[2] * (a[3] * a[7] - a[4] * a[6]);
    return det;
  } else {

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) ||                        \
    defined(USE_ACCELERATE_MPS)

    BLASINT *ipiv = (BLASINT *)malloc(mat->cols * sizeof(BLASINT));
    FloatMatrix *lu = sm_create_clone(mat);
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
    det = 1.0;
    for (size_t i = 0; i < mat->cols; i++) {
      det *= sm_get(lu, i, i);
    }
    free(ipiv);
    sm_destroy(lu);

#else
    FloatMatrix *copy = sm_create_clone(mat);
    if (!copy)
      return 0.0f;

    size_t *pivot_order = (size_t *)malloc(mat->rows * sizeof(size_t));
    if (!pivot_order) {
      sm_destroy(copy);
      return 0.0f;
    }

    if (!sm_lu_decompose(copy, pivot_order)) {
      free(pivot_order);
      sm_destroy(copy);
      return 0.0f;
    }

    float det = 1.0f;
    for (size_t i = 0; i < mat->rows; i++) {
      det *= copy->values[i * mat->cols + i];
      if (pivot_order[i] != i) {
        det *= -1.0f;
      }
    }

    free(pivot_order);
    sm_destroy(copy);
#endif
    return det;
  }
}

FloatMatrix *sm_inverse(const FloatMatrix *mat) {
  if (mat->cols != mat->rows || mat->rows == 0 || mat->cols == 0) {
    perror("the Matrix has to be square!");
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS) ||                        \
    defined(USE_ACCELERATE_MPS)
  FloatMatrix *inverse = sm_create_clone(mat);
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
  float *work = (float *)malloc(lwork * sizeof(float));
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
  FloatMatrix *copy = sm_create_clone(mat);
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
    for (ssize_t i = n - 1; i >= 0; i--) {
      float sum = inverse->values[i * n + col];
      for (size_t j = i + 1; j < n; j++) {
        sum -= copy->values[i * n + j] * inverse->values[j * n + col];
      }
      inverse->values[i * n + col] = sum / copy->values[i * n + i];
    }
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
  float norm = 0;
  for (size_t i = 0; i < mat->rows * mat->cols; i++) {
    norm += mat->values[i] * mat->values[i];
  }
  return sqrt(norm);
#endif
}

size_t sm_rank(const FloatMatrix *mat) {
  if (mat == NULL || mat->values == NULL) {
    return 0; // No matrix, no rank
  }
  int rank = 0;
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
  work = (float *)malloc(lwork * sizeof(float));
  if (work == NULL) {
    return -1; // Memory allocation failed
  }

  float *tau = (float *)malloc((m < n ? m : n) * sizeof(float));
  if (tau == NULL) {
    free(work);
    return -1; // Memory allocation failed
  }

  sgeqrf_(&m, &n, mat->values, &lda, tau, work, &lwork, &info);
  free(work);
  free(tau);
  if (info != 0) {
    return -1; // QR factorization failed
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

#pragma omp parallel for reduction(+ : counter)
  for (size_t i = 0; i < size; i++) {
    if (fabs(mat->values[i]) > EPSILON) {
      counter++;
    }
  }

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
  // Using Apple's Accelerate framework (= BLAS)
  cblas_saxpy((BLASINT)(mat1->rows * mat1->cols), 1.0, mat2->values, 1,
              mat1->values, 1);
#else
#pragma omp parallel for
  for (size_t i = 0; i < mat1->rows * mat1->cols; i++) {
    mat1->values[i] += mat2->values[i];
  }
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
#pragma omp parallel for
  for (size_t i = 0; i < mat1->rows * mat1->cols; i++) {
    mat1->values[i] -= mat2->values[i];
  }
#endif
}

void sm_inplace_square_transpose(FloatMatrix *mat) {
  if (mat == NULL || mat->values == NULL || mat->rows != mat->cols) {
    perror("Error: In-place transposition requires a square matrix.");
    return;
  }

  size_t n = mat->rows;

#pragma omp parallel for collapse(2) schedule(dynamic)
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
#pragma omp parallel for simd
  for (size_t i = 0; i < mat->rows * mat->cols; i++) {
    mat->values[i] *= scalar;
  }
#endif
}

FloatMatrix *sm_linear_batch(const FloatMatrix *inputs,
                             const FloatMatrix *weights,
                             const FloatMatrix *biases) {
  if (inputs == NULL || weights == NULL || biases == NULL) {
    perror("Error: Null pointer input to sm_linear_batch.\n");
    return NULL;
  }

  if (inputs->cols != weights->cols || biases->cols != weights->rows ||
      biases->rows != 1) {
    perror("Error: Dimension mismatch in sm_linear_batch.\n");
    return NULL;
  }

  FloatMatrix *output = sm_create(inputs->rows, weights->rows);
  if (!output)
    return NULL;

#ifdef USE_ACCELERATE
  // output = inputs * weights^T
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (BLASINT)inputs->rows,
              (BLASINT)weights->rows, (BLASINT)inputs->cols, 1.0f,
              inputs->values, (BLASINT)inputs->cols, weights->values,
              (BLASINT)weights->cols, 0.0f, output->values,
              (BLASINT)output->cols);

  // Add bias row-wise
  size_t rows = out->rows;
  size_t cols = out->cols;
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      out->values[i * cols + j] += bias->values[j];

#else

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < inputs->rows; i++) {
    for (size_t j = 0; j < weights->rows; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < inputs->cols; k++) {
        sum += inputs->values[i * inputs->cols + k] *
               weights->values[j * weights->cols + k];
      }
      output->values[i * output->cols + j] = sum + biases->values[j];
    }
  }
#endif

  return output;
}
