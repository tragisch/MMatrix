#include "dm.h"

#define INIT_CAPACITY 100
#define EPSILON 1e-9

// only for testing purposes:
// #ifdef __APPLE__
// #undef __APPLE__
// #endif

#ifdef __APPLE__
#define BLASINT int
#include <Accelerate/Accelerate.h>
#endif

/*******************************/
/*       Private Functions     */
/*******************************/

void dm_inplace_gauss_elimination(DoubleMatrix *mat) {
  size_t rows = mat->rows;
  size_t cols = mat->cols;

  for (size_t pivot = 0; pivot < rows; pivot++) {
    // Find the maximum value in the column below the pivot
    double max_val = fabs(dm_get(mat, pivot, pivot));
    size_t max_row = pivot;
    for (size_t row = pivot + 1; row < rows; row++) {
      double val = fabs(dm_get(mat, row, pivot));
      if (val > max_val) {
        max_val = val;
        max_row = row;
      }
    }

    // Swap rows if necessary
    if (max_row != pivot) {
      for (size_t col = 0; col < cols; col++) {
        double temp = dm_get(mat, pivot, col);
        dm_set(mat, pivot, col, dm_get(mat, max_row, col));
        dm_set(mat, max_row, col, temp);
      }
    }

    // Perform row operations to eliminate values below the pivot
    double pivot_val = dm_get(mat, pivot, pivot);
    if (fabs(pivot_val) > EPSILON) { // Check if pivot is non-zero
      for (size_t row = pivot + 1; row < rows; row++) {
        double factor = dm_get(mat, row, pivot) / pivot_val;
        dm_set(mat, row, pivot, 0.0); // Eliminate the value below the pivot

        for (size_t col = pivot + 1; col < cols; col++) {
          double val = dm_get(mat, row, col);
          dm_set(mat, row, col, val - factor * dm_get(mat, pivot, col));
        }
      }
    }
  }
}

size_t dm_rank_euler(const DoubleMatrix *mat) {
  // Make a copy of the matrix to preserve the original data
  DoubleMatrix *copy = dm_create(mat->rows, mat->cols);
  if (copy == NULL) {
    perror("Error: Memory allocation for matrix copy failed.\n");
    return 0; // Return 0 or an appropriate error value
  }
  memcpy(copy->values, mat->values, mat->rows * mat->cols * sizeof(double));

  // Apply Gaussian Elimination on the copy
  dm_inplace_gauss_elimination(copy);

  // Count the number of non-zero rows in the row-echelon form
  size_t rank = 0;
  for (size_t i = 0; i < copy->rows; i++) {
    int has_non_zero_element = 0;
    for (size_t j = 0; j < copy->cols; j++) {
      if (fabs(dm_get(copy, i, j)) > EPSILON) {
        has_non_zero_element = 1;
        break;
      }
    }
    if (has_non_zero_element) {
      rank++;
    }
  }

  // Free the memory of the copy
  dm_destroy(copy);

  return rank;
}

// Random number generation
double dm_rand_number() {
#ifdef __APPLE__
  uint32_t random_uint32 = arc4random();
#else
  uint32_t random_uint32 = rand();
#endif
  return (double)random_uint32 / (double)UINT32_MAX;
}

/*******************************/
/*      Public Functions      */
/*******************************/

DoubleMatrix *dm_create_empty() {
  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->rows = 0;
  matrix->cols = 0;
  matrix->capacity = 0;
  matrix->values = NULL;
  return matrix;
}

DoubleMatrix *dm_create_with_values(size_t rows, size_t cols, double *values) {
  DoubleMatrix *matrix = dm_create(rows, cols);
  matrix->cols = cols;
  matrix->rows = rows;
  matrix->capacity = rows * cols;
  matrix->values = values;
  return matrix;
}

DoubleMatrix *dm_create(size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->capacity = rows * cols;
  matrix->values = (double *)calloc(rows * cols, sizeof(double));
  return matrix;
}

DoubleMatrix *dm_create_clone(const DoubleMatrix *mat) {
  DoubleMatrix *copy = dm_create(mat->rows, mat->cols);
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(copy, i, j, dm_get(mat, i, j));
    }
  }
  return copy;
}

DoubleMatrix *dm_create_identity(size_t n) {
  DoubleMatrix *identity = dm_create(n, n);
  for (size_t i = 0; i < n; i++) {
    dm_set(identity, i, i, 1.0);
  }
  return identity;
}

double dm_rand_number();

DoubleMatrix *dm_create_random(size_t rows, size_t cols) {
  DoubleMatrix *mat = dm_create(rows, cols);

  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      double value = dm_rand_number();
      dm_set(mat, i, j, value);
    }
  }
  return mat;
}

DoubleMatrix *dm_create_from_array(size_t rows, size_t cols, double **array) {

  DoubleMatrix *mat = dm_create(rows, cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, array[i][j]);
    }
  }

  return mat;
}

DoubleMatrix *dm_create_from_2D_array(size_t rows, size_t cols,
                                      double array[rows][cols]) {
  DoubleMatrix *matrix = dm_create(rows, cols);
  if (!matrix)
    return NULL;

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      dm_set(matrix, i, j, array[i][j]);
    }
  }
  return matrix;
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
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  DoubleMatrix *product = dm_create(mat1->rows, mat2->cols);
#ifdef __APPLE__

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (BLASINT)mat1->rows,
              (BLASINT)mat2->cols, (BLASINT)mat1->cols, 1.0, mat1->values,
              (BLASINT)mat1->cols, mat2->values, (BLASINT)mat2->cols, 0.0,
              product->values, (BLASINT)product->cols);

#else

  for (size_t i = 0; i < mat1->rows; i++) {
    for (size_t j = 0; j < mat2->cols; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < mat1->cols; k++) {
        sum += dm_get(mat1, i, k) * dm_get(mat2, k, j);
      }
      dm_set(product, i, j, sum);
    }
  }

#endif
  return product;
}

DoubleMatrix *dm_multiply_by_number(const DoubleMatrix *mat,
                                    const double number) {
  DoubleMatrix *product = dm_create_clone(mat);
  dm_inplace_multiply_by_number(product, number);
  return product;
}

DoubleMatrix *dm_transpose(const DoubleMatrix *mat) {
  if (mat == NULL || mat->values == NULL)
    return NULL;

  DoubleMatrix *transposed = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  transposed->rows = mat->cols;
  transposed->cols = mat->rows;
  transposed->capacity = mat->cols * mat->rows;
  transposed->values = (double *)malloc(transposed->capacity * sizeof(double));
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(transposed, j, i, dm_get(mat, i, j));
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
  for (size_t i = 0; i < mat1->cols * mat1->rows; i++) {
    if (fabs(mat1->values[i] - mat2->values[i]) > EPSILON) {
      return false;
    }
  }
  return true;
}

DoubleMatrix *dm_add(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  DoubleMatrix *sum = dm_create_clone(mat1);
  dm_inplace_add(sum, mat2);

  return sum;
}

DoubleMatrix *dm_diff(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  DoubleMatrix *difference = dm_create_clone(mat1);
  dm_inplace_diff(difference, mat2);
  return difference;
}

double dm_determinant(const DoubleMatrix *mat) {
  if (mat->cols != mat->rows) {
    perror("the Matrix has to be square!");
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
#ifdef __APPLE__

    BLASINT *ipiv = (BLASINT *)malloc(mat->cols * sizeof(BLASINT));
    DoubleMatrix *lu = dm_create_clone(mat);
    BLASINT info = 0;
    BLASINT cols = (BLASINT)lu->cols;
    BLASINT rows = (BLASINT)lu->rows;

    dgetrf_(&cols, &rows, lu->values, &cols, ipiv, &info);
    if (info != 0) {
      perror("Error: dgetrf failed.\n");
      free(ipiv);
      dm_destroy(lu);
      return 0;
    }
    det = 1.0;
    for (size_t i = 0; i < mat->cols; i++) {
      det *= dm_get(lu, i, i);
    }
    free(ipiv);
    dm_destroy(lu);
#else
    double det = 0;
    for (size_t i = 0; i < mat->cols; i++) {
      DoubleMatrix *sub_mat = dm_create(mat->cols - 1, mat->cols - 1);
      for (size_t j = 1; j < mat->cols; j++) {
        for (size_t k = 0; k < mat->cols; k++) {
          if (k < i) {
            dm_set(sub_mat, j - 1, k, dm_get(mat, j, k));
          } else if (k > i) {
            dm_set(sub_mat, j - 1, k - 1, dm_get(mat, j, k));
          }
        }
      }
      det += pow(-1, (double)i) * dm_get(mat, 0, i) * dm_determinant(sub_mat);
      dm_destroy(sub_mat);
    }
#endif
    return det;
  }
}

DoubleMatrix *dm_inverse(const DoubleMatrix *mat) {
  if (mat->cols != mat->rows || mat->rows == 0 || mat->cols == 0) {
    perror("the Matrix has to be square!");
  }
  DoubleMatrix *inverse = dm_create_clone(mat);

#ifdef __APPLE__

  BLASINT *ipiv = (BLASINT *)malloc(mat->cols * sizeof(BLASINT));
  if (ipiv == NULL) {
    free(inverse->values);
    free(inverse);
    perror("Error: Memory allocation for ipiv failed.\n");
    return NULL;
  }
  BLASINT info = 0;
  BLASINT n = (BLASINT)inverse->cols;

  dgetrf_(&n, &n, inverse->values, &n, ipiv, &info);
  if (info != 0) {
    free(ipiv);
    free(inverse->values);
    free(inverse);
    perror("Error: dgetrf failed.\n");
    return NULL;
  }

  BLASINT lwork = -1;
  double work_opt;
  dgetri_(&n, inverse->values, &n, ipiv, &work_opt, &lwork, &info);

  lwork = (BLASINT)work_opt;
  double *work = (double *)malloc(lwork * sizeof(double));
  if (work == NULL) {
    free(ipiv);
    free(inverse->values);
    free(inverse);
    perror("Error: Memory allocation for work array failed.\n");
    return NULL;
  }

  dgetri_(&n, inverse->values, &n, ipiv, work, &lwork, &info);
  free(work);
  free(ipiv);
  if (info != 0) {
    free(inverse->values);
    free(inverse);
    perror("Error: dgetri failed.\n");
    return NULL;
  }
#else

  double det = dm_determinant(mat);
  if (fabs(det) < EPSILON) {
    perror("Error: Matrix is singular and cannot be inverted.\n");
    dm_destroy(inverse);
    return NULL;
  }

  for (size_t i = 0; i < mat->cols; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      DoubleMatrix *sub_mat = dm_create(mat->cols - 1, mat->cols - 1);
      if (sub_mat == NULL) {
        dm_destroy(inverse);
        perror("Error: Memory allocation for sub-matrix failed.\n");
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
  dm_inplace_multiply_by_number(inverse, 1 / det);
  dm_inplace_transpose(inverse);

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
  // allocate new memory for dense matrix:
  double *new_values = (double *)calloc(new_row * new_col, sizeof(double));

  if (new_values == NULL) {
    perror("Error: could not reallocate memory for dense matrix.\n");
    exit(EXIT_FAILURE);
  }

  // copy values from old matrix to new matrix:
  for (int i = 0; i < new_row; i++) {
    for (int j = 0; j < new_col; j++) {
      if (i >= mat->rows || j >= mat->cols) {
        new_values[i * new_col + j] = 0.0;
      } else {
        new_values[i * new_col + j] = mat->values[i * mat->cols + j];
      }
    }
  }

  // update matrix:
  mat->values = new_values;
  mat->rows = new_row;
  mat->cols = new_col;
  mat->capacity = new_row * new_col;
}

void dm_print(const DoubleMatrix *matrix) {
  for (size_t i = 0; i < matrix->rows; i++) {
    printf("[ ");
    for (size_t j = 0; j < matrix->cols; j++) {
      printf("%.2lf ", dm_get(matrix, i, j));
    }
    printf("]\n");
  }
  printf("Matrix (%zu x %zu)\n", matrix->rows, matrix->cols);
}

void dm_destroy(DoubleMatrix *mat) {
  free(mat->values);
  free(mat);
  mat = NULL;
}

double dm_trace(const DoubleMatrix *mat) {
#ifdef __APPLE__
  return cblas_ddot((BLASINT)(mat->rows), mat->values, (BLASINT)mat->cols + 1,
                    mat->values, (BLASINT)mat->cols + 1);
#else
  double trace = 0;
  for (size_t i = 0; i < mat->rows; i++) {
    trace += dm_get(mat, i, i);
  }
  return trace;
#endif
}

double dm_norm(const DoubleMatrix *mat) {
#ifdef __APPLE__
  return cblas_dnrm2((BLASINT)(mat->rows * mat->cols), mat->values, 1);
#else
  double norm = 0;
  for (size_t i = 0; i < mat->rows * mat->cols; i++) {
    norm += mat->values[i] * mat->values[i];
  }
  return sqrt(norm);
#endif
}

size_t dm_rank(const DoubleMatrix *mat) {
  if (mat == NULL || mat->values == NULL) {
    return 0; // No matrix, no rank
  }
  int rank = 0;
#ifdef __APPLE__
  BLASINT m = (BLASINT)mat->rows;
  BLASINT n = (BLASINT)mat->cols;
  BLASINT lda = n;
  BLASINT lwork = -1;
  double wkopt;
  double *work;
  BLASINT info;

  dgeqrf_(&m, &n, mat->values, &lda, NULL, &wkopt, &lwork, &info);
  lwork = (BLASINT)wkopt;
  work = (double *)malloc(lwork * sizeof(double));
  if (work == NULL) {
    return -1; // Memory allocation failed
  }

  double *tau = (double *)malloc((m < n ? m : n) * sizeof(double));
  if (tau == NULL) {
    free(work);
    return -1; // Memory allocation failed
  }

  dgeqrf_(&m, &n, mat->values, &lda, tau, work, &lwork, &info);
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
  rank = dm_rank_euler(mat);
#endif
  return rank;
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
void dm_inplace_add(DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
    perror("Error: invalid matrix dimensions.\n");
    return;
  }
#ifdef __APPLE__
  // Using Apple's Accelerate framework (= BLAS)
  cblas_daxpy((BLASINT)(mat1->rows * mat1->cols), 1.0, mat2->values, 1,
              mat1->values, 1);
#else
  for (size_t i = 0; i < mat1->rows; i++) {
    for (size_t j = 0; j < mat1->cols; j++) {
      dm_set(mat1, i, j, dm_get(mat1, i, j) + dm_get(mat2, i, j));
    }
  }
#endif
}

// In-place difference
void dm_inplace_diff(DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
    perror("Error: invalid matrix dimensions.\n");
    return;
  }
#ifdef __APPLE__
  // Using Apple's Accelerate framework (= BLAS)
  cblas_daxpy((BLASINT)(mat1->rows * mat1->cols), -1.0, mat2->values, 1,
              mat1->values, 1);
#else
  for (size_t i = 0; i < mat1->rows; i++) {
    for (size_t j = 0; j < mat1->cols; j++) {
      dm_set(mat1, i, j, dm_get(mat1, i, j) - dm_get(mat2, i, j));
    }
  }
#endif
}

// In-place transpose
void dm_inplace_transpose(DoubleMatrix *mat) {
  if (mat == NULL || mat->values == NULL || mat->rows != mat->cols) {
    perror("Error: In-place transposition requires a square matrix.");
    return;
  }

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = i + 1; j < mat->cols; j++) {

      double temp = dm_get(mat, i, j);
      dm_set(mat, i, j, dm_get(mat, j, i));
      dm_set(mat, j, i, temp);
    }
  }
}

// In-place scale
void dm_inplace_multiply_by_number(DoubleMatrix *mat, const double scalar) {
#ifdef __APPLE__
  cblas_dscal((BLASINT)(mat->rows * mat->cols), scalar, mat->values, 1);
#else
  for (size_t i = 0; i < mat->rows * mat->cols; i++) {
    mat->values[i] *= scalar;
  }
#endif
}
