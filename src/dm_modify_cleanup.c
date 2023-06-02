/**
 * @file dm_modify_cleanup.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief Clean up and drop drop small entries
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dbg.h"
#include "dm.h"
#include "dm_modify.h"
#include "dm_utils.h"
#include <math.h>

// TODO: use a better value for EPSILON
#define EPSILON 1e-10

/*******************************/
/*      drop small entries     */
/*******************************/

void dm_drop_small_entries(DoubleMatrix *mat) {
  switch (mat->format) {
  case COO:
    dm_drop_coo(mat);
    break;
  case CSC:
    break; // not implemented yet
  case DENSE:
    dm_drop_dense(mat);
    break;
  case VECTOR:
    break;
  }
}

static void dm_drop_coo(DoubleMatrix *mat) {
  // remove all entries with absolute value < EPSILON:
  for (int i = 0; i < mat->nnz; i++) {
    if (fabs(mat->values[i]) < EPSILON) {
      dm_remove_entry(mat, mat->row_indices[i], mat->col_indices[i]);
      i--;
    }
  }
}

static void dm_drop_dense(DoubleMatrix *mat) {
  // remove all entries with absolute value < EPSILON:
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      if (fabs(dm_get(mat, i, j)) < EPSILON) {
        dm_set(mat, i, j, 0.0);
      }
    }
  }
}

/*******************************/
/*       Order COO Entries     */
/*******************************/

// due to performance reason order the entries in the sparse matrix

void dm_order_coo(DoubleMatrix *mat) {
  // sort the entries in the sparse matrix:
  for (int i = 0; i < mat->nnz; i++) {
    for (int j = i + 1; j < mat->nnz; j++) {
      if (mat->row_indices[i] > mat->row_indices[j]) {
        dm_swap_entries_coo(mat, i, j);
      } else if (mat->row_indices[i] == mat->row_indices[j]) {
        if (mat->col_indices[i] > mat->col_indices[j]) {
          dm_swap_entries_coo(mat, i, j);
        }
      }
    }
  }
}

static void dm_swap_entries_coo(DoubleMatrix *mat, size_t i, size_t j) {
  size_t tmp_row = mat->row_indices[i];
  size_t tmp_col = mat->col_indices[i];
  double tmp_val = mat->values[i];
  mat->row_indices[i] = mat->row_indices[j];
  mat->col_indices[i] = mat->col_indices[j];
  mat->values[i] = mat->values[j];
  mat->row_indices[j] = tmp_row;
  mat->col_indices[j] = tmp_col;
  mat->values[j] = tmp_val;
}

/*******************************/
/*   Correct rows, cols (COO)  */
/*******************************/

static void dm_correct_rows_cols_coo(DoubleMatrix *mat) {
  // correct the rows and cols of the sparse matrix:
  size_t k_col = find_max_index(mat->col_indices, mat->nnz);
  size_t k_row = find_max_index(mat->row_indices, mat->nnz);

  if (k_col != mat->cols) {
    dbg(mat->cols = k_col); // Warning: this should not happen
  }

  if (k_row != mat->rows) {
    dbg(mat->rows = k_row); // Warning: this should not happen
  }
}
