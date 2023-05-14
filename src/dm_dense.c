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
  matrix->nnz = 0;
  matrix->format = DENSE;
  matrix->capacity = 0;
  matrix->row_indices = NULL;
  matrix->col_indices = NULL;
  matrix->values = (double *)calloc(rows * cols, sizeof(double));
  return matrix;
}


// create dense matrix from sparse matrix:
void dm_resize_dense(DoubleMatrix *mat, size_t new_row, size_t new_col) {

  // allocate new memory for dense matrix:
  double *new_values = (double *)calloc(new_row * new_col, sizeof(double));

  if (new_values == NULL) {
    perror("Error: could not reallocate memory for dense matrix.\n");
    exit(EXIT_FAILURE);
  }

  // copy values from old matrix to new matrix:
  for (int i = 0; i < new_row; i++) {
    for (int j = 0; j < new_col; j++) {
      new_values[i * new_col + j] = mat->values[i * mat->cols + j];
    }
  }

  // update matrix:
  mat->values = new_values;
  mat->rows = new_row;
  mat->cols = new_col;
  mat->capacity = new_row * new_col;
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

    // allocate memory for dense matrix:
    double *new_values =
        (double *)calloc(mat->rows * mat->cols, sizeof(double));
    size_t new_capacity = mat->rows * mat->cols;

    // fill dense matrix with values from sparse matrix:
    for (int i = 0; i < mat->nnz; i++) {
      new_values[mat->row_indices[i] * mat->cols + mat->col_indices[i]] =
          mat->values[i];
    }

    mat->format = DENSE;
    mat->values = new_values;
    mat->capacity = new_capacity;

    // free memory of sparse matrix:
    free(mat->row_indices);
    free(mat->col_indices);
    mat->row_indices = NULL;
    mat->col_indices = NULL;
  }
}
