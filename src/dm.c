#include "dm.h"
#include <stdlib.h>

#ifdef __APPLE__
#define BLASINT int
#include <Accelerate/Accelerate.h>
#endif

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

DoubleMatrix *dm_clone(const DoubleMatrix *mat) {
  DoubleMatrix *copy = dm_create(mat->rows, mat->cols);
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(copy, i, j, dm_get(mat, i, j));
    }
  }
  return copy;
}

DoubleMatrix *dm_identity(size_t n) {
  DoubleMatrix *identity = dm_create(n, n);
  for (size_t i = 0; i < n; i++) {
    dm_set(identity, i, i, 1.0);
  }
  return identity;
}

DoubleMatrix *dm_rand(size_t rows, size_t cols, double density) {
  DoubleMatrix *mat = dm_create(rows, cols);

  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
#ifdef __APPLE__
      if (arc4random() <= density) {
        double value = arc4random_uniform(100) / 100.0;
#else
      if (rand() <= density) {
        double value = rand() % 100 / 100.0;
#endif
        dm_set(mat, i, j, value);
      }
    }
  }
  return mat;
}

// DoubleMatrix *dm_convert_array(size_t rows, size_t cols,
//                                double array[rows][cols]) {
//   DoubleMatrix *mat = dm_create(rows, cols);

//   for (size_t i = 0; i < mat->rows; i++) {
//     for (size_t j = 0; j < mat->cols; j++) {
//       dm_set(mat, i, j, array[i][j]);
//     }
//   }

//   return mat;
// }

DoubleMatrix *dm_import_array(size_t rows, size_t cols, double **array) {
  DoubleMatrix *mat = dm_create(rows, cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, array[i][j]);
    }
  }

  return mat;
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
  // Using Apple's Accelerate framework (= BLAS)
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
  DoubleMatrix *product = dm_clone(mat);
  dm_multiply_me_by_number(product, number);
  return product;
}

void dm_multiply_me_by_number(DoubleMatrix *mat, const double number) {
#ifdef __APPLE__
  // Using Apple's Accelerate framework (= BLAS)
  cblas_dscal((BLASINT)(mat->rows * mat->cols), number, mat->values, 1);

#else

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      mat->values[i * mat->cols + j] *= number;
    }
  }

#endif
}

DoubleMatrix *dm_transpose(const DoubleMatrix *mat) {
  if (mat == NULL || mat->values == NULL)
    return NULL;

  DoubleMatrix *transposed = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  if (transposed == NULL)
    return NULL;

  transposed->rows = mat->cols;
  transposed->cols = mat->rows;
  transposed->values =
      (double *)malloc(transposed->rows * transposed->cols * sizeof(double));
  if (transposed->values == NULL) {
    free(transposed);
    return NULL;
  }

  for (int i = 0; i < mat->rows; ++i) {
    for (int j = 0; j < mat->cols; ++j) {
      transposed->values[j * transposed->cols + i] =
          mat->values[i * mat->cols + j];
    }
  }

  return transposed;
}

bool dm_equal(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
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
  DoubleMatrix *sum = dm_clone(mat1);

#ifdef __APPLE__
  // Using Apple's Accelerate framework (= BLAS)
  cblas_daxpy((BLASINT)(mat1->rows * mat1->cols), 1.0, mat2->values, 1,
              sum->values, 1);

#else

  for (size_t i = 0; i < mat1->rows; i++) {
    for (size_t j = 0; j < mat1->cols; j++) {
      dm_set(sum, i, j, dm_get(mat1, i, j) + dm_get(mat2, i, j));
    }
  }
#endif
  return sum;
}

DoubleMatrix *dm_diff(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  DoubleMatrix *difference = dm_clone(mat1);

#ifdef __APPLE__

  // Using Apple's Accelerate framework (= BLAS)
  cblas_daxpy((BLASINT)(mat1->rows * mat1->cols), -1.0, mat2->values, 1,
              difference->values, 1);

#else

  for (size_t i = 0; i < mat1->rows; i++) {
    for (size_t j = 0; j < mat1->cols; j++) {
      dm_set(difference, i, j, dm_get(mat1, i, j) - dm_get(mat2, i, j));
    }
  }

#endif
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
    // Using Apple's Accelerate framework (= BLAS)
    int *ipiv = (int *)malloc(mat->cols * sizeof(int));
    DoubleMatrix *lu = dm_clone(mat);
    int info = 0;
    dgetrf_((BLASINT *)&lu->cols, (BLASINT *)&lu->rows, lu->values,
            (BLASINT *)&lu->cols, ipiv, (BLASINT *)&info);
    if (info != 0) {
      perror("Error: dgetrf failed.\n");
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

  if (mat->cols != mat->rows) {
    perror("the Matrix has to be square!");
  }
  DoubleMatrix *inverse = dm_clone(mat);

#ifdef __APPLE__
  // Using Apple's Accelerate framework (= BLAS)
  int *ipiv = (int *)malloc(mat->cols * sizeof(int));
  if (ipiv == NULL) {
    free(inverse->values);
    free(inverse);
    perror("Error: Memory allocation for ipiv failed.\n");
    return NULL;
  }
  int info = 0;
  int n = (int)inverse->cols;

  // LU decomposition
  dgetrf_(&n, &n, inverse->values, &n, ipiv, &info);
  if (info != 0) {
    free(ipiv);
    free(inverse->values);
    free(inverse);
    perror("Error: dgetrf failed.\n");
    return NULL;
  }

  // Query the optimal size of the work array
  int lwork = -1;
  double work_opt;
  dgetri_(&n, inverse->values, &n, ipiv, &work_opt, &lwork, &info);

  lwork = (int)work_opt;
  double *work = (double *)malloc(lwork * sizeof(double));
  if (work == NULL) {
    free(ipiv);
    free(inverse->values);
    free(inverse);
    perror("Error: Memory allocation for work array failed.\n");
    return NULL;
  }

  // Matrix inversion
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

  for (size_t i = 0; i < mat->cols; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      DoubleMatrix *sub_mat = dm_create(mat->cols - 1, mat->cols - 1);
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
      dm_set(inverse, i, j, pow(-1, (double)(i + j)) * dm_determinant(sub_mat));
      dm_destroy(sub_mat);
    }
  }
  dm_transpose(inverse);
  dm_multiply_by_scalar(inverse, 1 / det);
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
  int m = (int)mat->rows;
  int n = (int)mat->cols;
  int k = (m < n) ? m : n;
  int lda = n;
  int lwork = -1;
  double wkopt;
  double *work;
  int info;

  dgeqrf_(&m, &n, mat->values, &lda, NULL, &wkopt, &lwork, &info);
  lwork = (int)wkopt;
  work = (double *)malloc(lwork * sizeof(double));
  if (work == NULL) {
    return -1; // Memory allocation failed
  }

  double *tau = (double *)malloc(k * sizeof(double));
  if (tau == NULL) {
    free(work);
    return -1; // Memory allocation failed
  }

  double *a_copy = (double *)malloc(mat->rows * mat->cols * sizeof(double));
  if (a_copy == NULL) {
    free(work);
    free(tau);
    return -1; // Memory allocation failed
  }
  memcpy(a_copy, mat->values, mat->rows * mat->cols * sizeof(double));
  dgeqrf_(&m, &n, a_copy, &lda, tau, work, &lwork, &info);

  free(work);
  free(tau);
  if (info != 0) {
    free(a_copy);
    return -1; // QR factorization failed
  }

  for (int i = 0; i < k; ++i) {
    if (fabs(a_copy[i * lda + i]) > EPSILON) {
      rank++;
    }
  }
  free(a_copy);
#else
  rank = dm_rank_euler(mat);
#endif
  return rank;
}

static size_t dm_rank_euler(const DoubleMatrix *mat) {
  // Make a copy of the matrix to preserve the original data
  DoubleMatrix *copy = dm_clone(mat);

  // Apply Gaussian Elimination on the copy
  dm_gauss_elimination(copy);

  // Count the number of non-zero rows in the row-echelon form
  size_t rank = 0;
  for (size_t i = 0; i < copy->rows; i++) {
    int non_zero = 0;
    for (size_t j = 0; j < copy->cols; j++) {
      if (dm_get(copy, i, j) != 0.0) {
        non_zero = 1;
        break;
      }
    }
    if (non_zero) {
      rank++;
    }
  }

  // Free the memory of the copy
  dm_destroy(copy);

  return rank;
}

// Gaussian Elimination
static void dm_gauss_elimination(DoubleMatrix *mat) {
  size_t rows = mat->rows;
  size_t cols = mat->cols;

  // Apply Gaussian Elimination
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
    for (size_t row = pivot + 1; row < rows; row++) {
      double factor = dm_get(mat, row, pivot) / dm_get(mat, pivot, pivot);
      dm_set(mat, row, pivot, 0.0); // Eliminate the value below the pivot

      for (size_t col = pivot + 1; col < cols; col++) {
        double val = dm_get(mat, row, col);
        double pivot_val = dm_get(mat, pivot, col);
        dm_set(mat, row, col, val - factor * pivot_val);
      }
    }
  }
}

double ur_rand() {
#ifdef __APPLE__
  uint32_t random_uint32 = arc4random();
#else
  uint32_t random_uint32 = rand();
#endif
  return (double)random_uint32 / (double)UINT32_MAX;
}
