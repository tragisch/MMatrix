/**
 * @file dm_modify_resize.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm.h"
#include "dm_modify.h"

/*******************************/
/*         Resize Matrix       */
/*******************************/

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
  case COO:
    dm_resize_coo(mat, new_row, new_col);
    break;
  case CSC:
    dm_resize_csc(mat, new_row, new_col);
    break; // not implemented yet
  case VECTOR:
    dm_resize_dense(mat, new_row, 1);
    break;
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

// resize matrix of COO format:
static void dm_resize_coo(DoubleMatrix *mat, size_t new_row, size_t new_col) {

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

// resize matrix of CSC format:
static void dm_resize_csc(DoubleMatrix *mat, size_t new_rows, size_t new_cols) {
  if (mat == NULL || mat->format != CSC) {
    // Invalid matrix or incorrect format
    return;
  }

  // Resize row_indices and values arrays
  size_t nnz = mat->col_ptrs[mat->cols] - mat->col_ptrs[0];

  // realloc
  size_t *new_row_indices = realloc(mat->row_indices, nnz * sizeof(size_t));
  double *new_values = realloc(mat->values, nnz * sizeof(double));

  if (new_row_indices == NULL || new_values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  mat->row_indices = new_row_indices;
  mat->values = new_values;

  // Update matrix properties
  mat->rows = new_rows;
  mat->cols = new_cols;

  // Resize col_ptrs array
  size_t col_ptrs_size = (new_cols + 1) * sizeof(size_t);
  size_t *new_col_ptrs = realloc(mat->col_ptrs, col_ptrs_size);
  if (new_col_ptrs == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  mat->col_ptrs = new_col_ptrs;

  // Adjust col_ptrs values based on the new matrix size
  double scale_factor = (double)new_rows / (double)mat->rows;
  for (size_t i = 0; i <= new_cols; i++) {
    mat->col_ptrs[i] = (size_t)((double)mat->col_ptrs[i] * scale_factor);
  }
}