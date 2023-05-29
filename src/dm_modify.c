/**
 * @file dm_modify_.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm_modify.h"
#include "dm.h"
#include "dm_vector.h"

/*******************************/
/*      Get Row and Column     */
/*******************************/

/**
 * @brief Get the Row Vector object of Row row
 *
 * @param mat
 * @param row
 * @return DoubleVector
 */
DoubleVector *dm_get_row(DoubleMatrix *mat, size_t row_idx) {
  if (row_idx < 0 || row_idx > (mat->rows - 1)) {
    perror("This row does not exist");
  }
  DoubleVector *vec = dv_create(mat->cols);
  if (mat->format == DENSE) {
    memcpy(vec->values, mat->values + row_idx * mat->cols,
           mat->cols * sizeof(double));
  }
  for (size_t i = 0; i < vec->rows; i++) {
    dv_set(vec, i, dm_get(mat, row_idx, i));
  }

  return vec;
}

/**
 * @brief Get the Column Vector object
 *
 * @param mat
 * @param column
 * @return DoubleVector
 */
DoubleVector *dm_get_column(DoubleMatrix *mat, size_t column_idx) {
  if (column_idx < 0 || column_idx > (mat->cols) - 1) {
    perror("This column does not exist");
  }
  DoubleVector *vec = dv_create(mat->rows);
  for (size_t i = 0; i < mat->rows; i++) {
    dv_set(vec, i, dm_get(mat, i, column_idx));
  }

  return vec;
}

/*******************************/
/*     Set Row and Column     */
/*******************************/

/**
 * @brief Set the Column object at index column_idx
 *
 * @param mat
 * @param column_idx
 * @param vec
 */
void dm_set_column(DoubleMatrix *mat, size_t column_idx, DoubleVector *vec) {
  if (vec->rows != mat->rows) {
    perror("Error: Length of vector does not fit to number or matrix rows");
  } else {
    for (size_t i = 0; i < mat->rows; i++) {
      dm_set(mat, i, column_idx, dv_get(vec, i));
    }
  }
}

/**
 * @brief Set the Row object at index row_idx
 *
 * @param mat
 * @param row_idx
 * @param vec
 */
void dm_set_row(DoubleMatrix *mat, size_t row_idx, DoubleVector *vec) {
  if (vec->rows != mat->cols) {
    perror("Error: Length of vector does not fit to number or matrix columns");
  } else {
    for (size_t i = 0; i < mat->cols; i++) {
      dm_set(mat, row_idx, i, dv_get(vec, i));
    }
  }
}

/*******************************/
/*          Sub-Matrix         */
/*******************************/

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
  DoubleMatrix *sub_mat = dm_create_format(
      row_end - row_start + 1, col_end - col_start + 1, mat->format);
  for (size_t i = row_start; i <= row_end; i++) {
    for (size_t j = col_start; j <= col_end; j++) {
      dm_set(sub_mat, i - row_start, j - col_start, dm_get(mat, i, j));
    }
  }
  return sub_mat;
}
