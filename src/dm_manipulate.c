/**
 * @file dm_manipulate.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dbg.h"
#include "dm.h"
#include "dm_internals.h"
#include "dm_math.h"
#include "dv_vector.h"

enum { INIT_CAPACITY = 2U };

/*******************************/
/*      Manipulate Matrix      */
/*******************************/

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
    size_t new_rows = mat->rows + 1;
    dm_resize(mat, new_rows, mat->cols);

    for (size_t i = 0; i < mat->cols; i++) {
      dm_set(mat, new_rows - 1, i, dv_get(row_vec, i));
    }
  }
}

void dm_convert(DoubleMatrix *mat, matrix_format format) {
  if (mat->format == format) {
    return;
  }
  switch (format) {
  case DENSE:
    dm_convert_to_dense(mat);

    break;
  case SPARSE:
    dm_convert_to_sparse(mat);
    break;

  case VECTOR:
    break;
  }
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
    exit(EXIT_FAILURE);
  }
  if (row_start > row_end || col_start > col_end) {
    perror("Error: matrix index out of bounds.\n");
    exit(EXIT_FAILURE);
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

/**
 * @brief resize matrix to new size
 *
 * @param mat
 * @param new_row
 * @param new_col
 */
void dm_resize(DoubleMatrix *mat, size_t new_row, size_t new_col) {
  switch (mat->format) {
  case DENSE:
    dm_resize_dense(mat, new_row, new_col);
    break;
  case SPARSE:
    dm_resize_sparse(mat, new_row, new_col);
    break;
  case VECTOR:
    dm_resize_dense(mat, new_row, 1);
    break;
  default:
    break;
  }
}

/*******************************/
/*      private functions      */
/*******************************/

// convert dense matrix to sparse matrix of COO format:
static void dm_convert_to_sparse(DoubleMatrix *mat) {
  // check if matrix is already in sparse format:
  if (mat->format == SPARSE) {
    printf("Matrix is already in sparse format!\n");
    return;
  }

  // convert matrix:
  size_t nnz = 0;
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      if (is_zero(mat->values[i * mat->cols + j]) == false) {
        nnz++;
      }
    }
  }

  // allocate memory for sparse matrix:
  size_t *row_indices = (size_t *)calloc(nnz + 1, sizeof(size_t));
  size_t *col_indices = (size_t *)calloc(nnz + 1, sizeof(size_t));
  double *values = (double *)calloc(nnz + 1, sizeof(double));
  if (row_indices == NULL || col_indices == NULL || values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  // fill sparse matrix:
  size_t k = 0;
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      if (is_zero(mat->values[i * mat->cols + j]) == false) {
        row_indices[k] = i;
        col_indices[k] = j;
        values[k] = mat->values[i * mat->cols + j];
        k++;
      }
    }
  }

  // free memory of dense matrix:
  free(mat->values);

  // set sparse matrix:
  mat->format = SPARSE;
  mat->nnz = nnz;
  mat->capacity = nnz;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
}

// convert SparseMatrix of COO format to Dense format
static void dm_convert_to_dense(DoubleMatrix *mat) {
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

// create dense matrix from sparse matrix:
static void dm_resize_dense(DoubleMatrix *mat, size_t new_row, size_t new_col) {

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

// resize matrix of COO format:
static void dm_resize_sparse(DoubleMatrix *mat, size_t new_row,
                             size_t new_col) {
  // check if matrix is already in sparse format:
  if (mat->format != SPARSE) {
    printf("Can not resize matrix to sparse format!\n");
    exit(EXIT_FAILURE);
  }

  // resize matrix:
  size_t *row_indices =
      (size_t *)realloc(mat->row_indices, new_row * new_col * sizeof(size_t));
  size_t *col_indices =
      (size_t *)realloc(mat->col_indices, new_row * new_col * sizeof(size_t));
  double *values =
      (double *)realloc(mat->values, new_row * new_col * sizeof(double));
  if (row_indices == NULL || col_indices == NULL || values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  mat->capacity = new_row * new_col;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
  mat->rows = new_row;
  mat->cols = new_col;
}
