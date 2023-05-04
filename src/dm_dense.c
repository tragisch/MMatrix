/**
 * @file dm_dense_matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dbg.h"
#include "dm_io.h"
#include "dm_math.h"
#include "dm_matrix.h"

/**
 * @brief Create a zero Double Matrix object
 *
 * @param rows
 * @param cols
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_dense(size_t rows, size_t cols) {

  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->format = DENSE;
  matrix->values =
      (double *)malloc((matrix->rows * matrix->cols) * sizeof(double));
  return matrix;
}

// change size of dense matrix:
void dm_resize_dense(DoubleMatrix *mat, size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    perror("Matrix dimensions must be greater than 0");
  } else {
    // in case of a dense matrix:
    if (mat->format == DENSE) {
      double *new_data = (double *)calloc(rows * cols, sizeof(double));
      size_t min_rows = mat->rows < rows ? mat->rows : rows;
      size_t min_cols = mat->cols < cols ? mat->cols : cols;
      for (size_t i = 0; i < min_rows; i++) {
        for (size_t j = 0; j < min_cols; j++) {
          new_data[i * cols + j] = mat->values[i * mat->cols + j];
        }
      }
      free(mat->values);
      mat->values = new_data;
      mat->rows = rows;
      mat->cols = cols;
    }
  }
}

// get value from dense matrix:
void dm_set_dense(DoubleMatrix *mat, size_t i, size_t j, const double value) {
  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return;
  }
  mat->values[i * mat->cols + j] = value;
  mat->nnz++;
}

// get value from dense matrix:
double dm_get_dense(const DoubleMatrix *mat, size_t i, size_t j) {
  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return 0.0;
  }
  return mat->values[i * mat->cols + j];
}

// convert SparseMatrix of CSR format to Dense format
void dm_convert_to_dense(DoubleMatrix *mat) {
  if (mat->format == SPARSE) {
    // create new dense matrix:
    DoubleMatrix *new_mat = dm_create_dense(mat->rows, mat->cols);
    if (new_mat == NULL) {
      perror("Error: could not create new dense matrix.\n");
      dm_destroy(mat);
    } else {
      // fill new dense matrix with values from sparse matrix:
      for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
          new_mat->values[i * mat->cols + j] = dm_get_sparse(mat, i, j);
        }
      }
      // free sparse matrix:
      free(mat->values);

      // copy new dense matrix to old sparse matrix:
      mat->rows = new_mat->rows;
      mat->cols = new_mat->cols;
      mat->format = new_mat->format;
      mat->values = new_mat->values;
      mat->nnz = new_mat->nnz;
      mat->row_pointers = new_mat->row_pointers;
      mat->col_indices = new_mat->col_indices;
      free(new_mat);
    }
  } else {
    perror("Error: matrix format is not sparse (CSR).\n");
  }
}
