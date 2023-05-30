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
    dm_resize_sparse(mat, new_row, new_col);
    break;
  case CSR:
    break; // not implemented yet
  case VECTOR:
    dm_resize_dense(mat, new_row, 1);
    break;
  case HASHTABLE:
    dm_resize_hastable(mat, new_row, new_col);
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

static void dm_resize_hastable(DoubleMatrix *mat, size_t new_row,
                               size_t new_col) {
  // resize hash table:
  kh_resize(entry, mat->hash_table, new_row * new_col);
  if (mat->hash_table == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }
  mat->rows = new_row;
  mat->cols = new_col;
}
