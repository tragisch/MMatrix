/**
 * @file dm_matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm_matrix.h"

#include <assert.h>

#include "dbg.h"

// #define NDEBUG
enum { INIT_CAPACITY = 2U };

/*******************************/
/*        Double Matrix        */
/*******************************/

/**
 * @brief create an empty Double Matrix Object
 *
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_matrix() {
  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->cols = 0U;
  matrix->rows = 0U;
  matrix->is_sparse = false;
  matrix->values = (double *)malloc(INIT_CAPACITY * sizeof(double));
  return matrix;
}

/**
 * @brief Create a zero Double Matrix object
 *
 * @param rows
 * @param cols
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create(size_t rows, size_t cols) {

  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->rows = rows;
  matrix->cols = cols;

  matrix->values =
      (double *)malloc((matrix->rows * matrix->cols) * sizeof(double));
  return matrix;
}

/**
 * @brief Create a Random Double Matrix object
 *
 * @param num_rows
 * @param num_cols
 * @return double**
 */
DoubleMatrix *dm_create_rand(size_t rows, size_t cols) {
  DoubleMatrix *mat = dm_create(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      dm_set(mat, i, j, randomDouble());
    }
  }

  return mat;
}

/**
 * @brief Create a Identity object
 *
 * @param rows
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_identity(size_t rows) {
  DoubleMatrix *mat = dm_create(rows, rows);
  for (size_t i = 0; i < rows; i++) {
    dm_set(mat, i, i, 1);
  }
  return mat;
}

/**
 * @brief Set the Array To Matrix object
 *
 * @param rows
 * @param cols
 * @param array
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols,
                                   double array[rows][cols]) {
  DoubleMatrix *mat = dm_create(rows, cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, array[i][j]);
    }
  }

  return mat;
}

/**
 * @brief return copy of matrix
 *
 * @param m
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_clone(DoubleMatrix *mat) {
  DoubleMatrix *copy = dm_create(mat->rows, mat->cols);
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(copy, i, j, dm_get(mat, i, j));
    }
  }
  return copy;
}

bool dm_is_vector(DoubleMatrix *mat) {
  if (mat->rows == 1 || mat->cols == 1) {
    return true; // Matrix is a vector
  }
  return false; // Matrix is not a vector
}

/**
 * @brief Resize the matrix
 *
 * @param rows
 * @param cols
 * @param mat
 */

void dm_resize(DoubleMatrix *mat, size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    perror("destroy matrix instead of setting zero sizes!");
  } else {
    // in case of a dense matrix:
    if ((mat->cols != cols) || (mat->rows != rows)) {
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

/**
 * @brief push (add) a column vector to  matrix
 *
 * @param mat
 * @param col_vec
 */
void dm_push_column(DoubleMatrix *mat, DoubleVector *col_vec) {
  if (mat->rows != col_vec->rows) {
    perror("Error: Length of vector does not fit to number or matrix rows");
  } else {
    // resize the matrix:
    dm_resize(mat, mat->rows, mat->cols + 1);
    for (size_t i = 0; i < mat->rows; i++) {
      dm_set(mat, i, mat->cols - 1, dv_get(col_vec, i));
    }
  }
}

/**
 * @brief push (add) a row vector to matrix
 *
 * @param mat
 * @param row_vec
 */
void dm_push_row(DoubleMatrix *mat, DoubleVector *row_vec) {
  if (row_vec->rows != mat->cols) {
    perror("Error: length of vector does not fit to number or matrix columns");

  } else {
    // resize the matrix:
    dm_resize(mat, mat->rows + 1, mat->cols);
    for (size_t i = 0; i < mat->cols; i++) {
      dm_set(mat, mat->rows - 1, i, dv_get(row_vec, i));
    }
  }
}

/**
 * @brief get value of index i, j
 *
 * @param mat
 * @param i,j
 * @return double
 */
double dm_get(const DoubleMatrix *mat, size_t i, size_t j) {

  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return 0;
  }
  if (mat->is_sparse) {
    return sp_get(mat, i, j);
  }
  return mat->values[i * mat->cols + j];
}

/**
 * @brief set value of index i, j
 *
 * @param mat
 * @param i,j
 * @param value
 */
void dm_set(DoubleMatrix *mat, size_t i, size_t j, const double value) {
  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return;
  }
  if (mat->is_sparse) {
    sp_set(mat, i, j, value);
  } else {
    mat->values[i * mat->cols + j] = value;
  }
}

/**
 * @brief free memory of DoubleMatrix
 *
 * @param mat
 */
void dm_destroy(DoubleMatrix *mat) { sp_destroy(mat); }

/**
 * @brief convert sparse matrix to dense matrix
 *
 * @param mat
 */
DoubleMatrix *dm_sparse_to_dense(SparseMatrix *sp_mat) {
  DoubleMatrix *mat = dm_create(sp_mat->rows, sp_mat->cols);
  for (size_t i = 0; i < sp_mat->rows; i++) {
    for (size_t j = 0; j < sp_mat->cols; j++) {
      dm_set(mat, i, j, sp_get(sp_mat, i, j));
    }
  }
  return mat;
}

/**
 * @brief get sub matrix of matrix
 *
 * @param mat
 * @param row_start
 * @param row_end
 * @param col_start
 * @param col_end
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_get_sub_matrix(DoubleMatrix *mat, size_t row_start,
                                size_t row_end, size_t col_start,
                                size_t col_end) {
  if (row_start < 0 || row_start > mat->rows || row_end < 0 ||
      row_end > mat->rows || col_start < 0 || col_start > mat->cols ||
      col_end < 0 || col_end > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return NULL;
  }
  if (row_start > row_end || col_start > col_end) {
    perror("Error: matrix index out of bounds.\n");
    return NULL;
  }
  DoubleMatrix *sub_mat =
      dm_create(row_end - row_start + 1, col_end - col_start + 1);
  for (size_t i = row_start; i <= row_end; i++) {
    for (size_t j = col_start; j <= col_end; j++) {
      dm_set(sub_mat, i - row_start, j - col_start, dm_get(mat, i, j));
    }
  }
  return sub_mat;
}
