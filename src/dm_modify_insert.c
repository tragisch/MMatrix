/**
 * @file dm_modify_insert.c
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
#include "dm_vector.h"

/*******************************/
/*        Insert Column        */
/*******************************/

void dm_insert_column(DoubleMatrix *mat, size_t column_idx, DoubleVector *vec) {
  if (vec->rows != mat->rows) {
    perror("Error: Length of vector does not fit to number or matrix rows");
  } else {
    switch (mat->format) {
    case DENSE:
      dm_insert_column_dense(mat, column_idx);
      break;
    case COO:
      dm_insert_column_coo(mat, column_idx);
      break;
    case CSR:
      break; // not implemented yet
    case VECTOR:
      break;
    }

    // insert the new column:
    for (size_t i = 0; i < mat->rows; i++) {
      dm_set(mat, i, column_idx, dv_get(vec, i));
    }
  }
}

static void dm_insert_column_coo(DoubleMatrix *mat, size_t column_idx) {
  // resize the matrix:
  dm_resize(mat, mat->rows, mat->cols + 1);

  // shift all columns to the right:
  for (size_t i = 0; i < mat->nnz; i++) {
    if (mat->col_indices[i] >= (column_idx)) {
      mat->col_indices[i]++;
    }
  }
}

static void dm_insert_column_dense(DoubleMatrix *mat, size_t column_idx) {
  // resize the matrix:
  mat->cols++;

  // allocate new memory for the values:
  double *value = realloc(mat->values, sizeof(double) * mat->rows * mat->cols);

  if (value == NULL) {
    perror("Error: Could not allocate memory for new column");
    exit(EXIT_FAILURE);
  }

  // Reallocate memory for the updated values array
  mat->values = value;

  // Shift existing values to the right from the insertion position
  for (size_t row = 0; row < mat->rows; row++) {
    for (size_t col = mat->cols - 1; col > column_idx; col--) {
      size_t newIndex = row * mat->cols + col;
      size_t oldIndex = row * (mat->cols - 1) + (col - 1);
      mat->values[newIndex] = mat->values[oldIndex];
    }
  }
}



/*******************************/
/*         Insert Row          */
/*******************************/

void dm_insert_row(DoubleMatrix *mat, size_t row_idx, DoubleVector *vec) {
  if (vec->rows != mat->cols) {
    perror("Error: Length of vector does not fit to number or matrix columns");
  } else {
    switch (mat->format) {
    case DENSE:
      dm_insert_row_dense(mat, row_idx);
      break;
    case COO:
      dm_insert_row_coo(mat, row_idx);
      break;
    case CSR:
      break; // not implemented yet
    case VECTOR:
      break;
    }

    // insert the new row:
    for (size_t i = 0; i < mat->cols; i++) {
      dm_set(mat, row_idx, i, dv_get(vec, i));
    }
  }
}

static void dm_insert_row_coo(DoubleMatrix *mat, size_t row_idx) {
  // resize the matrix:
  dm_resize(mat, mat->rows + 1, mat->cols);

  // shift all rows to the bottom:
  for (size_t i = 0; i < mat->nnz; i++) {
    if (mat->row_indices[i] >= (row_idx)) {
      mat->row_indices[i]++;
    }
  }
}

static void dm_insert_row_dense(DoubleMatrix *mat, size_t row_idx) {
  // resize the matrix:
  mat->rows++;

  // allocate new memory for the values:
  double *value = realloc(mat->values, sizeof(double) * mat->rows * mat->cols);

  if (value == NULL) {
    perror("Error: Could not allocate memory for new row");
    exit(EXIT_FAILURE);
  }

  // Reallocate memory for the updated values array
  mat->values = value;

  // Shift existing values to the bottom from the insertion position
  for (size_t col = 0; col < mat->cols; col++) {
    for (size_t row = mat->rows - 1; row > row_idx; row--) {
      size_t newIndex = row * mat->cols + col;
      size_t oldIndex = (row - 1) * mat->cols + col;
      mat->values[newIndex] = mat->values[oldIndex];
    }
  }
}

