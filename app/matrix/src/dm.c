/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "dm.h"

#include <log.h>
#include <omp.h>
#include <pcg_variants.h>
#include <stdint.h>

#define INIT_CAPACITY 100
static const double EPSILON = 1e-9;
static uint64_t dm_global_seed = 0;
static bool dm_seed_initialized = false;

/*******************************/
/*      Define Environment     */
/*******************************/

/*******************************/
/*      Define Environment     */
/*******************************/

#if defined(USE_ACCELERATE)
#define ACTIVE_LIB "Apple Accelerate"
#elif defined(USE_OPENBLAS)
#define ACTIVE_LIB "OpenBLAS"
#else
#define ACTIVE_LIB "No BLAS"
#endif

#ifdef USE_ACCELERATE
#define BLASINT int
#include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
#define BLASINT int
#include <cblas.h>
#include <lapacke.h>
#else
#include <math.h>
#include <omp.h>
#endif

#define BLOCK_SIZE 64

/*******************************/
/*       Private Functions     */
/*******************************/

bool dm_inplace_gauss_elimination(DoubleMatrix *mat) {
  if (!mat || !mat->values || mat->rows == 0 || mat->cols == 0) {
    log_error("Error: invalid matrix for Gaussian elimination.\n");
    return false;
  }

  size_t rows = mat->rows;
  size_t cols = mat->cols;

  double *A = mat->values;
  for (size_t pivot = 0; pivot < rows; pivot++) {
    // Find the maximum value in the column below the pivot
    double max_val = fabs(A[pivot * cols + pivot]);
    size_t max_row = pivot;
    for (size_t row = pivot + 1; row < rows; row++) {
      double val = fabs(A[row * cols + pivot]);
      if (val > max_val) {
        max_val = val;
        max_row = row;
      }
    }

    // Swap rows if necessary
    if (max_row != pivot) {
      for (size_t col = 0; col < cols; col++) {
        double temp = A[pivot * cols + col];
        A[pivot * cols + col] = A[max_row * cols + col];
        A[max_row * cols + col] = temp;
      }
    }

    // Perform row operations to eliminate values below the pivot
    double pivot_val = dm_get(mat, pivot, pivot);
    if (fabs(pivot_val) > EPSILON) {  // Check if pivot is non-zero
      double *A = mat->values;
      size_t ld = mat->cols;

#pragma omp parallel for
      for (size_t row = pivot + 1; row < rows; row++) {
        double factor = A[row * ld + pivot] / pivot_val;
        A[row * ld + pivot] = 0.0;

#pragma omp simd
        for (size_t col = pivot + 1; col < cols; col++) {
          A[row * ld + col] -= factor * A[pivot * ld + col];
        }
      }
    }
  }

  return true;
}

size_t dm_rank_euler(const DoubleMatrix *mat) {
  // Make a copy of the matrix to preserve the original data
  DoubleMatrix *copy = dm_create(mat->rows, mat->cols);
  if (copy == NULL) {
    log_error("Error: Memory allocation for matrix copy failed.\n");
    return 0;  // Return 0 or an appropriate error value
  }
  memcpy(copy->values, mat->values, mat->rows * mat->cols * sizeof(double));

  // Apply Gaussian Elimination on the copy
  if (!dm_inplace_gauss_elimination(copy)) {
    dm_destroy(copy);
    return 0;
  }

  // Count the number of non-zero rows in the row-echelon form
  size_t rank = 0;
#pragma omp parallel for reduction(+ : rank)
  for (size_t i = 0; i < copy->rows; i++) {
    const double *row = &copy->values[i * copy->cols];
    int has_nonzero = 0;
#pragma omp simd reduction(| : has_nonzero)
    for (size_t j = 0; j < copy->cols; j++) {
      has_nonzero |= (fabs(row[j]) > EPSILON);
    }
    rank += (size_t)has_nonzero;
  }

  // Free the memory of the copy
  dm_destroy(copy);

  return rank;
}

// Convert DoubleMatrix to column-major double array
double *dm_to_column_major(const DoubleMatrix *mat) {
  if (!mat || !mat->values) {
    log_error("Error: matrix is NULL.\n");
    return NULL;
  }

  size_t rows = mat->rows;
  size_t cols = mat->cols;
  double *col_major = (double *)malloc(rows * cols * sizeof(double));
  if (!col_major) {
    log_error("Error: could not allocate column-major array.\n");
    return NULL;
  }

#pragma omp parallel for
  for (size_t i = 0; i < rows; ++i) {
#pragma omp simd
    for (size_t j = 0; j < cols; ++j) {
      col_major[j * rows + i] = mat->values[i * cols + j];
    }
  }

  return col_major;
}

static uint64_t dm_mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static uint64_t dm_resolve_seed(uint64_t seed) {
  if (seed != 0) {
    return seed;
  }
  if (dm_seed_initialized) {
    return dm_global_seed;
  }
  return ((uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)&dm_global_seed);
}

void dm_set_random_seed(uint64_t seed) {
  dm_global_seed = seed;
  dm_seed_initialized = true;
}

uint64_t dm_get_random_seed(void) {
  if (!dm_seed_initialized) {
    return 0;
  }
  return dm_global_seed;
}

/*******************************/
/*      Public Functions      */
/*******************************/

const char *dm_active_library(void) { return ACTIVE_LIB; }


DoubleMatrix *dm_create_empty(void) {
  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->rows = 0;
  matrix->cols = 0;
  matrix->capacity = 0;
  matrix->values = NULL;
  return matrix;
}

DoubleMatrix *dm_create_with_values(size_t rows, size_t cols, double *values) {
  DoubleMatrix *matrix = dm_create(rows, cols);
  if (matrix == NULL) {
    return NULL;
  }

  if (values != NULL) {
    memcpy(matrix->values, values, rows * cols * sizeof(double));
  }

  return matrix;
}

DoubleMatrix *dm_create(size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->capacity = rows * cols;
  matrix->values = (double *)calloc(rows * cols, sizeof(double));
  return matrix;
}

DoubleMatrix *dm_clone(const DoubleMatrix *mat) {
  DoubleMatrix *copy = dm_create(mat->rows, mat->cols);
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(copy, i, j, dm_get(mat, i, j));
    }
  }
  return copy;
}

DoubleMatrix *dm_create_clone(const DoubleMatrix *mat) { return dm_clone(mat); }

DoubleMatrix *dm_create_identity(size_t n) {
  DoubleMatrix *identity = dm_create(n, n);
  for (size_t i = 0; i < n; i++) {
    dm_set(identity, i, i, 1.0);
  }
  return identity;
}

DoubleMatrix *dm_create_random_seeded(size_t rows, size_t cols, uint64_t seed) {
  DoubleMatrix *mat = dm_create(rows, cols);
  if (!mat) {
    return NULL;
  }
  size_t total = rows * cols;
  uint64_t base_seed = dm_resolve_seed(seed);

  const double inv_u53 = 1.0 / 9007199254740992.0;  // 2^53
#pragma omp parallel for
  for (size_t i = 0; i < total; ++i) {
    uint64_t mixed = dm_mix64(base_seed ^ ((uint64_t)i * 0x9E3779B97F4A7C15ull));
    mat->values[i] = (double)(mixed >> 11) * inv_u53;
  }

  return mat;
}

DoubleMatrix *dm_create_random(size_t rows, size_t cols) {
  return dm_create_random_seeded(rows, cols, 0);
}

DoubleMatrix *dm_from_array_ptrs(size_t rows, size_t cols, double **array) {
  DoubleMatrix *mat = dm_create(rows, cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, array[i][j]);
    }
  }

  return mat;
}

DoubleMatrix *dm_create_from_array(size_t rows, size_t cols, double **array) {
  return dm_from_array_ptrs(rows, cols, array);
}

DoubleMatrix *dm_from_array_static(size_t rows, size_t cols,
                                   double array[rows][cols]) {
  DoubleMatrix *matrix = dm_create(rows, cols);
  if (!matrix) return NULL;

  double *dst = matrix->values;

#pragma omp parallel for
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      dst[i * cols + j] = array[i][j];
    }
  }
  return matrix;
}

DoubleMatrix *dm_create_from_2D_array(size_t rows, size_t cols,
                                      double array[rows][cols]) {
  return dm_from_array_static(rows, cols, array);
}

DoubleMatrix *dm_get_row(const DoubleMatrix *mat, size_t i) {
  DoubleMatrix *row = dm_create(1, mat->cols);
  for (size_t j = 0; j < mat->cols; j++) {
    dm_set(row, 0, j, dm_get(mat, i, j));
  }
  return row;
}

DoubleMatrix *dm_get_last_row(const DoubleMatrix *mat) {
  return dm_get_row(mat, mat->rows - 1);
}

DoubleMatrix *dm_get_col(const DoubleMatrix *mat, size_t j) {
  DoubleMatrix *col = dm_create(mat->rows, 1);
  for (size_t i = 0; i < mat->rows; i++) {
    dm_set(col, i, 0, dm_get(mat, i, j));
  }
  return col;
}

DoubleMatrix *dm_get_last_col(const DoubleMatrix *mat) {
  return dm_get_col(mat, mat->cols - 1);
}

DoubleMatrix *dm_multiply(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1->cols != mat2->rows) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  DoubleMatrix *product = dm_create(mat1->rows, mat2->cols);
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (BLASINT)mat1->rows,
              (BLASINT)mat2->cols, (BLASINT)mat1->cols, 1.0, mat1->values,
              (BLASINT)mat1->cols, mat2->values, (BLASINT)mat2->cols, 0.0,
              product->values, (BLASINT)product->cols);

#else

  // Fallback to manual multiplication
  DoubleMatrix *mat2_transposed = dm_transpose(mat2);

  double *a = mat1->values;
  double *bT = mat2_transposed->values;
  double *c = product->values;
  size_t rows = mat1->rows;
  size_t cols = mat2->cols;
  size_t inner = mat1->cols;

#pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      double sum = 0.0;
#pragma omp simd reduction(+ : sum)
      for (size_t k = 0; k < inner; k++) {
        sum += a[i * inner + k] * bT[j * inner + k];
      }
      c[i * cols + j] = sum;
    }
  }
  dm_destroy(mat2_transposed);

#endif
  return product;
}

DoubleMatrix *dm_multiply_by_number(const DoubleMatrix *mat,
                                    const double number) {
  DoubleMatrix *product = dm_clone(mat);
  if (!product) return NULL;
  if (!dm_inplace_multiply_by_number(product, number)) {
    dm_destroy(product);
    return NULL;
  }
  return product;
}

DoubleMatrix *dm_transpose(const DoubleMatrix *mat) {
  if (mat == NULL || mat->values == NULL) return NULL;

  size_t rows = mat->rows;
  size_t cols = mat->cols;
  DoubleMatrix *transposed = dm_create(cols, rows);

  double *src = mat->values;
  double *dst = transposed->values;

#pragma omp parallel for collapse(2)
  for (size_t ii = 0; ii < rows; ii += BLOCK_SIZE) {
    for (size_t jj = 0; jj < cols; jj += BLOCK_SIZE) {
      for (size_t i = ii; i < ii + BLOCK_SIZE && i < rows; i++) {
        for (size_t j = jj; j < jj + BLOCK_SIZE && j < cols; j++) {
          dst[j * rows + i] = src[i * cols + j];
        }
      }
    }
  }

  return transposed;
}

bool dm_is_equal(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1 == NULL || mat2 == NULL) {
    return false;
  }
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    return false;
  }

  const double *a = mat1->values;
  const double *b = mat2->values;
  size_t size = mat1->cols * mat1->rows;

  unsigned int equal = 1;

#pragma omp simd reduction(& : equal)
  for (size_t i = 0; i < size; ++i) {
    equal &= (fabs(a[i] - b[i]) <= EPSILON);
  }

  return equal;
}

DoubleMatrix *dm_add(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }

  DoubleMatrix *result = dm_clone(mat1);
  if (!result) return NULL;

  double *a = result->values;
  const double *b = mat2->values;
  size_t size = result->rows * result->cols;

#if defined(USE_ACCELERATE)
  vDSP_vaddD(a, 1, b, 1, a, 1, size);
#elif defined(USE_OPENBLAS)
  for (size_t i = 0; i < size; ++i) {
    a[i] += b[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    a[i] += b[i];
  }
#endif

  return result;
}

DoubleMatrix *dm_diff(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }

  DoubleMatrix *result = dm_clone(mat1);
  if (!result) return NULL;

  double *a = result->values;
  const double *b = mat2->values;
  size_t size = result->rows * result->cols;

#if defined(USE_ACCELERATE)
  vDSP_vsubD(b, 1, a, 1, a, 1, size);  // result = a - b
#elif defined(USE_OPENBLAS)
  for (size_t i = 0; i < size; ++i) {
    a[i] -= b[i];
  }
#else
#pragma omp simd
  for (size_t i = 0; i < size; ++i) {
    a[i] -= b[i];
  }
#endif

  return result;
}

double dm_determinant(const DoubleMatrix *mat) {
  if (mat->cols != mat->rows) {
    log_error("the Matrix has to be square!");
  }
  double det = 0;
  if (mat->cols == 1) {
    return dm_get(mat, 0, 0);
  } else if (mat->cols == 2) {
    return dm_get(mat, 0, 0) * dm_get(mat, 1, 1) -
           dm_get(mat, 0, 1) * dm_get(mat, 1, 0);
  } else if (mat->cols == 3) {
    double *a = mat->values;
    det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
          a[1] * (a[3] * a[8] - a[5] * a[6]) +
          a[2] * (a[3] * a[7] - a[4] * a[6]);
    return det;
  } else {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)

    BLASINT *ipiv = (BLASINT *)malloc(mat->cols * sizeof(BLASINT));
    DoubleMatrix *lu = dm_clone(mat);
    BLASINT info = 0;
    BLASINT cols = (BLASINT)lu->cols;
    BLASINT rows = (BLASINT)lu->rows;

    dgetrf_(&cols, &rows, lu->values, &cols, ipiv, &info);
    if (info != 0) {
      log_error("Error: dgetrf failed.\n");
      free(ipiv);
      dm_destroy(lu);
      return 0;
    }
    // det = 1.0;
    double local_det = 1.0;
#pragma omp parallel
    {
      double thread_det = 1.0;
      double *values = lu->values;
      size_t stride = lu->cols + 1;
      size_t n = mat->cols;
#pragma omp simd reduction(* : thread_det)
      for (size_t i = 0; i < n; i++) {
        thread_det *= values[i * stride];
      }
#pragma omp critical
      {
        local_det *= thread_det;
      }
    }
    det = local_det;
    free(ipiv);
    dm_destroy(lu);

#else
    // Optimized fallback with direct memory access and OpenMP
    double det = 0.0;
    size_t n = mat->cols;
    const double *src = mat->values;

#pragma omp parallel for reduction(+ : det)
    for (size_t i = 0; i < n; i++) {
      DoubleMatrix *sub_mat = dm_create(n - 1, n - 1);
      double *dst = sub_mat->values;

      for (size_t j = 1; j < n; j++) {
        for (size_t k = 0; k < n; k++) {
          if (k < i) {
            dst[(j - 1) * (n - 1) + k] = src[j * n + k];
          } else if (k > i) {
            dst[(j - 1) * (n - 1) + (k - 1)] = src[j * n + k];
          }
        }
      }

      double sign = (i % 2 == 0) ? 1.0 : -1.0;
      det += sign * src[i] * dm_determinant(sub_mat);
      dm_destroy(sub_mat);
    }
#endif
    return det;
  }
}

DoubleMatrix *dm_inverse(const DoubleMatrix *mat) {
  if (mat->cols != mat->rows || mat->rows == 0 || mat->cols == 0) {
    log_error("the Matrix has to be square!");
  }
  DoubleMatrix *inverse = dm_clone(mat);

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)

  BLASINT *ipiv = (BLASINT *)malloc(mat->cols * sizeof(BLASINT));
  if (ipiv == NULL) {
    free(inverse->values);
    free(inverse);
    log_error("Error: Memory allocation for ipiv failed.\n");
    return NULL;
  }
  BLASINT info = 0;
  BLASINT n = (BLASINT)inverse->cols;

  dgetrf_(&n, &n, inverse->values, &n, ipiv, &info);
  if (info != 0) {
    free(ipiv);
    free(inverse->values);
    free(inverse);
    log_error("Error: dgetrf failed.\n");
    return NULL;
  }

  BLASINT lwork = -1;
  double work_opt;
  dgetri_(&n, inverse->values, &n, ipiv, &work_opt, &lwork, &info);

  lwork = (BLASINT)work_opt;
  double *work = (double *)malloc((size_t)lwork * sizeof(double));
  if (work == NULL) {
    free(ipiv);
    free(inverse->values);
    free(inverse);
    log_error("Error: Memory allocation for work array failed.\n");
    return NULL;
  }

  dgetri_(&n, inverse->values, &n, ipiv, work, &lwork, &info);
  free(work);
  free(ipiv);
  if (info != 0) {
    free(inverse->values);
    free(inverse);
    log_error("Error: dgetri failed.\n");
    return NULL;
  }
#else

  double det = dm_determinant(mat);
  if (fabs(det) < EPSILON) {
    log_error("Error: Matrix is singular and cannot be inverted.\n");
    dm_destroy(inverse);
    return NULL;
  }

  for (size_t i = 0; i < mat->cols; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      DoubleMatrix *sub_mat = dm_create(mat->cols - 1, mat->cols - 1);
      if (sub_mat == NULL) {
        dm_destroy(inverse);
        log_error("Error: Memory allocation for sub-matrix failed.\n");
        return NULL;
      }

      // Erstellen der Untermatrix
      for (size_t k = 0; k < mat->cols; k++) {
        for (size_t l = 0; l < mat->cols; l++) {
          if (k < i && l < j) {
            dm_set(sub_mat, k, l, dm_get(mat, k, l));
          } else if (k < i && l > j) {
            dm_set(sub_mat, k, l - 1, dm_get(mat, k, l));
          } else if (k > i && l < j) {
            dm_set(sub_mat, k - 1, l, dm_get(mat, k, l));
          } else if (k > i && l > j) {
            dm_set(sub_mat, k - 1, l - 1, dm_get(mat, k, l));
          }
        }
      }

      double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
      dm_set(inverse, i, j, sign * dm_determinant(sub_mat));

      dm_destroy(sub_mat);
    }
  }

  // Skalieren mit 1 / det
  if (!dm_inplace_multiply_by_number(inverse, 1 / det) ||
      !dm_inplace_transpose(inverse)) {
    dm_destroy(inverse);
    return NULL;
  }

#endif
  return inverse;
}

void dm_set(DoubleMatrix *mat, size_t i, size_t j, const double value) {
  mat->values[i * mat->cols + j] = value;
}

double dm_get(const DoubleMatrix *mat, size_t i, size_t j) {
  return mat->values[i * mat->cols + j];
}

void dm_reshape(DoubleMatrix *matrix, size_t new_rows, size_t new_cols) {
  matrix->rows = new_rows;
  matrix->cols = new_cols;
}

void dm_resize(DoubleMatrix *mat, size_t new_row, size_t new_col) {
  double *old_values = mat->values;

  // allocate new memory for dense matrix:
  double *new_values = (double *)calloc(new_row * new_col, sizeof(double));

  if (new_values == NULL) {
    log_error("Error: could not reallocate memory for dense matrix.\n");
    exit(EXIT_FAILURE);
  }

  // copy values from old matrix to new matrix, OpenMP-parallelized:
  size_t min_rows = (new_row < mat->rows) ? new_row : mat->rows;
  size_t min_cols = (new_col < mat->cols) ? new_col : mat->cols;

#pragma omp parallel for
  for (size_t i = 0; i < min_rows; i++) {
#pragma omp simd
    for (size_t j = 0; j < min_cols; j++) {
      new_values[i * new_col + j] = mat->values[i * mat->cols + j];
    }
  }

  // update matrix:
  mat->values = new_values;
  mat->rows = new_row;
  mat->cols = new_col;
  mat->capacity = new_row * new_col;

  free(old_values);
}

void dm_print(const DoubleMatrix *matrix) {
  size_t rows = matrix->rows;
  size_t cols = matrix->cols;
  const double *values = matrix->values;

  for (size_t i = 0; i < rows; i++) {
    printf("[ ");
    for (size_t j = 0; j < cols; j++) {
      printf("%.3lf ", values[i * cols + j]);
    }
    printf("]\n");
  }
  printf("Matrix[Double.3f] (%zu x %zu)\n", rows, cols);
}

void dm_destroy(DoubleMatrix *mat) {
  free(mat->values);
  free(mat);
  mat = NULL;
}

double dm_trace(const DoubleMatrix *mat) {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  return cblas_ddot((BLASINT)(mat->rows), mat->values, (BLASINT)mat->cols + 1,
                    mat->values, (BLASINT)mat->cols + 1);
#else
  double trace = 0;
  size_t size = mat->rows;
  const double *values = mat->values;
  size_t stride = mat->cols + 1;

#pragma omp simd reduction(+ : trace)
  for (size_t i = 0; i < size; i++) {
    trace += values[i * stride];
  }
  return trace;
#endif
}

double dm_norm(const DoubleMatrix *mat) {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  return cblas_dnrm2((BLASINT)(mat->rows * mat->cols), mat->values, 1);
#else
  double norm = 0;
  const double *values = mat->values;
  size_t size = mat->rows * mat->cols;

#pragma omp simd reduction(+ : norm)
  for (size_t i = 0; i < size; i++) {
    norm += values[i] * values[i];
  }

  return sqrt(norm);
#endif
}

size_t dm_rank(const DoubleMatrix *mat) {
  if (mat == NULL || mat->values == NULL) {
    return 0;  // No matrix, no rank
  }
  size_t rank = 0;
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  DoubleMatrix *tmp = dm_clone(mat);
  if (tmp == NULL) {
    return 0;
  }

  BLASINT m = (BLASINT)mat->rows;
  BLASINT n = (BLASINT)mat->cols;
  BLASINT lda = n;
  BLASINT lwork = -1;
  double wkopt;
  double *work;
  BLASINT info;

  dgeqrf_(&m, &n, tmp->values, &lda, NULL, &wkopt, &lwork, &info);
  lwork = (BLASINT)wkopt;
  work = (double *)malloc((size_t)lwork * sizeof(double));
  if (work == NULL) {
    dm_destroy(tmp);
    return 0;  // Memory allocation failed
  }

  double *tau = (double *)malloc((size_t)((m < n) ? m : n) * sizeof(double));
  if (tau == NULL) {
    free(work);
    dm_destroy(tmp);
    return 0;  // Memory allocation failed
  }

  dgeqrf_(&m, &n, tmp->values, &lda, tau, work, &lwork, &info);
  free(work);
  free(tau);
  if (info != 0) {
    dm_destroy(tmp);
    return 0;  // QR factorization failed
  }

  int k = (m < n) ? m : n;
  for (int i = 0; i < k; ++i) {
    if (fabs(tmp->values[i * lda + i]) > EPSILON) {
      rank++;
    }
  }

  dm_destroy(tmp);
#else
  rank = dm_rank_euler(mat);
#endif
  return (size_t)rank;
}

double dm_density(const DoubleMatrix *mat) {
  int counter = 0;
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      if (fabs(dm_get(mat, i, j)) > EPSILON) {
        counter++;
      }
    }
  }
  return (double)counter / (double)(mat->rows * mat->cols);
}

// Matrix is empty
bool dm_is_empty(const DoubleMatrix *mat) {
  return (mat == NULL || mat->values == NULL || mat->rows == 0 ||
          mat->cols == 0);
}

// Matrix is square
bool dm_is_square(const DoubleMatrix *mat) { return (mat->rows == mat->cols); }

// Matrix is vector
bool dm_is_vector(const DoubleMatrix *mat) {
  return (mat->rows == 1 || mat->cols == 1);
}

// Matrix is equal size
bool dm_is_equal_size(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  return (mat1->rows == mat2->rows && mat1->cols == mat2->cols);
}

// In-place operations
bool dm_inplace_add(DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (!mat1 || !mat2 || !mat1->values || !mat2->values) {
    log_error("Error: invalid matrix input.\n");
    return false;
  }

  if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
    log_error("Error: invalid matrix dimensions.\n");
    return false;
  }

  double *a = mat1->values;
  const double *b = mat2->values;
  size_t size = mat1->rows * mat1->cols;

#if defined(USE_ACCELERATE)
  vDSP_vaddD(a, 1, b, 1, a, 1, size);
#elif defined(USE_OPENBLAS)
  for (size_t i = 0; i < size; ++i) {
    a[i] += b[i];
  }
#else
#pragma omp simd
  for (size_t i = 0; i < size; ++i) {
    a[i] += b[i];
  }
#endif

  return true;
}

// In-place difference
bool dm_inplace_diff(DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (!mat1 || !mat2 || !mat1->values || !mat2->values) {
    log_error("Error: invalid matrix input.\n");
    return false;
  }

  if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
    log_error("Error: invalid matrix dimensions.\n");
    return false;
  }

  double *a = mat1->values;
  const double *b = mat2->values;
  size_t size = mat1->rows * mat1->cols;

#if defined(USE_ACCELERATE)
  vDSP_vsubD(b, 1, a, 1, a, 1, size);  // result = a - b
#elif defined(USE_OPENBLAS)
  for (size_t i = 0; i < size; ++i) {
    a[i] -= b[i];
  }
#else
#pragma omp simd
  for (size_t i = 0; i < size; ++i) {
    a[i] -= b[i];
  }
#endif

  return true;
}

// In-place transpose
bool dm_inplace_transpose(DoubleMatrix *mat) {
  if (mat == NULL || mat->values == NULL || mat->rows != mat->cols) {
    log_error("Error: In-place transposition requires a square matrix.");
    return false;
  }

  double *A = mat->values;
  size_t n = mat->cols;  // matrix is square

  for (size_t i = 0; i < n; i++) {
#pragma omp simd
    for (size_t j = i + 1; j < n; j++) {
      double temp = A[i * n + j];
      A[i * n + j] = A[j * n + i];
      A[j * n + i] = temp;
    }
  }

  return true;
}

bool dm_inplace_multiply_by_number(DoubleMatrix *mat, const double scalar) {
  if (!mat || !mat->values) {
    log_error("Error: invalid matrix input.\n");
    return false;
  }

  double *a = mat->values;
  size_t size = mat->rows * mat->cols;

#if defined(USE_ACCELERATE)
  vDSP_vsmulD(a, 1, &scalar, a, 1, size);
#elif defined(USE_OPENBLAS)
  cblas_dscal((BLASINT)size, scalar, a, 1);
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < size; ++i) {
    a[i] *= scalar;
  }
#endif

  return true;
}

DoubleMatrix *dm_elementwise_multiply(const DoubleMatrix *mat1,
                                      const DoubleMatrix *mat2) {
  if (!mat1 || !mat2 || !dm_is_equal_size(mat1, mat2)) {
    log_error("Error: invalid matrix dimensions for Hadamard product.\n");
    return NULL;
  }

  DoubleMatrix *result = dm_clone(mat1);
  if (!result) return NULL;

  double *a = result->values;
  const double *b = mat2->values;
  size_t size = result->rows * result->cols;

#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    a[i] *= b[i];
  }

  return result;
}

// Elementwise division (SIMD-optimized)
DoubleMatrix *dm_div(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (!mat1 || !mat2 || !dm_is_equal_size(mat1, mat2)) {
    log_error("Error: invalid matrix dimensions for elementwise division.\n");
    return NULL;
  }

  DoubleMatrix *result = dm_clone(mat1);
  if (!result) return NULL;

  double *a = result->values;
  const double *b = mat2->values;
  size_t size = result->rows * result->cols;

#if defined(USE_ACCELERATE)
  vDSP_vdivD(b, 1, a, 1, a, 1, size);
#elif defined(USE_OPENBLAS)
  for (size_t i = 0; i < size; ++i) {
    a[i] /= b[i];
  }
#else
#pragma omp simd
  for (size_t i = 0; i < size; ++i) {
    a[i] /= b[i];
  }
#endif

  return result;
}

bool dm_inplace_elementwise_multiply(DoubleMatrix *mat1,
                                     const DoubleMatrix *mat2) {
  if (!mat1 || !mat2 || !dm_is_equal_size(mat1, mat2)) {
    log_error("Error: invalid matrix dimensions for Hadamard product.\n");
    return false;
  }

  double *a = mat1->values;
  const double *b = mat2->values;
  size_t size = mat1->rows * mat1->cols;

#if defined(USE_ACCELERATE)
  vDSP_vmulD(a, 1, b, 1, a, 1, size);
#elif defined(USE_OPENBLAS)
  for (size_t i = 0; i < size; ++i) {
    a[i] *= b[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < size; ++i) {
    a[i] *= b[i];
  }
#endif

  return true;
}

bool dm_inplace_div(DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (!mat1 || !mat2 || !dm_is_equal_size(mat1, mat2)) {
    log_error("Error: invalid matrix dimensions for elementwise division.\n");
    return false;
  }

  double *a = mat1->values;
  const double *b = mat2->values;
  size_t size = mat1->rows * mat1->cols;

#if defined(USE_ACCELERATE)
  vDSP_vdivD(b, 1, a, 1, a, 1, size);
#elif defined(USE_OPENBLAS)
  for (size_t i = 0; i < size; ++i) {
    a[i] /= b[i];
  }
#else
#pragma omp simd
  for (size_t i = 0; i < size; ++i) {
    a[i] /= b[i];
  }
#endif

  return true;
}
