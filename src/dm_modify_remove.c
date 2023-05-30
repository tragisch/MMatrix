/**
 * @file dm_modify_remove.c
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
/*       Remove  entry         */
/*******************************/

void dm_remove_entry(DoubleMatrix *mat, size_t i, size_t j) {
  switch (mat->format) {
  case COO:
    dm_remove_entry_sparse(mat, i, j);
    break;
  case DENSE:
    break; // nothing to do
  case HASHTABLE:
    dm_remove_entry_hashtable(mat, i, j);
    break;
  case VECTOR:
    break; // nothing to do
  }
}

// remove nnz value at index i,j of sparse matrix in COO format:
static void dm_remove_entry_sparse(DoubleMatrix *mat, size_t i, size_t j) {
  for (int k = 0; k < mat->nnz; k++) {
    if ((mat->row_indices[k] == i) && (mat->col_indices[k] == j)) {
      // remove element if found
      mat->nnz--;
      for (int l = k; l < mat->nnz; l++) {
        mat->row_indices[l] = mat->row_indices[l + 1];
        mat->col_indices[l] = mat->col_indices[l + 1];
        mat->values[l] = mat->values[l + 1];
      }
      return;
    }
  }
}

// remove entry from at index i,j of hash table matrix:
static void dm_remove_entry_hashtable(DoubleMatrix *mat, size_t i, size_t j) {
  int64_t key = ((int64_t)i << 32) | (int64_t)j;
  khiter_t iter = kh_get(entry, mat->hash_table, key);
  if (iter != kh_end(mat->hash_table)) {
    kh_del(entry, mat->hash_table, iter);
    mat->nnz--;
  }
}

/*******************************/
/*         Remove Column       */
/*******************************/

void dm_remove_column(DoubleMatrix *mat, size_t column_idx) {
  if (column_idx < 0 || column_idx > (mat->cols - 1)) {
    perror("This column does not exist");
  }
  switch (mat->format) {
  case DENSE:
    dm_remove_column_dense(mat, column_idx);
    break;
  case COO:
    dm_remove_column_sparse(mat, column_idx);
    break;
  case HASHTABLE:
    dm_remove_column_hashtable(mat, column_idx);
    break;
  case VECTOR:
    break;
  }
}

static void dm_remove_column_sparse(DoubleMatrix *mat, size_t column_idx) {
  // shift all columns to the left:
  for (size_t i = 0; i < mat->nnz; i++) {
    if (mat->col_indices[i] == column_idx - 1) {
      mat->nnz--;
    }
    if (mat->col_indices[i] >= column_idx) {
      mat->col_indices[i]--;
    }
  }

  // resize the matrix:
  dm_resize(mat, mat->rows, mat->cols - 1);
}

static void dm_remove_column_dense(DoubleMatrix *mat, size_t column_idx) {
  // shift all columns to the left:
  for (size_t i = column_idx; i < mat->cols - 1; i++) {
    for (size_t j = 0; j < mat->rows; j++) {
      dm_set(mat, j, i, dm_get(mat, j, i + 1));
    }
  }

  // resize the matrix:
  dm_resize(mat, mat->rows, mat->cols - 1);
}

static void dm_remove_column_hashtable(DoubleMatrix *mat, size_t column_idx) {
  // Create a new hash table for the updated matrix
  khash_t(entry) *new_hash_table = kh_init(entry);
  kh_resize(entry, new_hash_table, mat->rows * (mat->cols - 1));

  // Iterate over the existing entries in the hash table
  for (khiter_t iter = kh_begin(mat->hash_table);
       iter != kh_end(mat->hash_table); ++iter) {
    if (kh_exist(mat->hash_table, iter)) {
      int64_t key = kh_key(mat->hash_table, iter);
      double value = kh_value(mat->hash_table, iter);
      size_t row = (size_t)(key >> 32);
      size_t col = (size_t)(key & 0xFFFFFFFF);

      if (col != column_idx) {
        if (col > column_idx) {
          col--;
        }

        int64_t new_key = ((int64_t)row << 32) | (int64_t)col;
        int ret = 0;
        khiter_t new_iter = kh_put(entry, new_hash_table, new_key, &ret);
        kh_value(new_hash_table, new_iter) = value;
      }
    }
  }

  // Free the old hash table
  kh_destroy(entry, mat->hash_table);

  // Update the hash table pointer in the matrix
  mat->hash_table = new_hash_table;
  mat->cols--;
}

/*******************************/
/*         Remove Row          */
/*******************************/

void dm_remove_row(DoubleMatrix *mat, size_t row_idx) {
  if (row_idx < 0 || row_idx > (mat->rows - 1)) {
    perror("This row does not exist");
  }
  switch (mat->format) {
  case DENSE:
    dm_remove_row_dense(mat, row_idx);
    break;
  case COO:
    dm_remove_row_sparse(mat, row_idx);
    break;
  case HASHTABLE:
    dm_remove_row_hashtable(mat, row_idx);
    break;
  case VECTOR:
    break;
  }
}

static void dm_remove_row_sparse(DoubleMatrix *mat, size_t row_idx) {
  size_t *temp_row_index = calloc(mat->nnz, sizeof(size_t));
  size_t *temp_col_index = calloc(mat->nnz, sizeof(size_t));
  double *temp_values = calloc(mat->nnz, sizeof(double));
  size_t new_nnz = 0;

  // copy all values except the ones in the row to be removed:
  int k = 0;
  for (size_t i = 0; i < mat->nnz; i++) {
    if (mat->row_indices[i] != row_idx) {
      temp_row_index[k] = mat->row_indices[i];
      temp_col_index[k] = mat->col_indices[i];
      temp_values[k] = mat->values[i];
      new_nnz++;
      k++;
    }
  }

  // shift all rows to the top:
  for (size_t i = 0; i < mat->nnz; i++) {
    if (temp_row_index[i] >= row_idx) {
      temp_row_index[i]--;
    }
  }

  // copy the values back:
  free(mat->row_indices);
  free(mat->col_indices);
  free(mat->values);
  mat->row_indices = temp_row_index;
  mat->col_indices = temp_col_index;
  mat->values = temp_values;
  mat->rows--;
  mat->nnz = new_nnz;
}

static void dm_remove_row_dense(DoubleMatrix *mat, size_t row_idx) {
  // shift all rows to the top:
  for (size_t i = row_idx; i < mat->rows - 1; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, dm_get(mat, i + 1, j));
    }
  }

  // resize the matrix:
  dm_resize(mat, mat->rows - 1, mat->cols);
}

static void dm_remove_row_hashtable(DoubleMatrix *mat, size_t row_idx) {
  // Create a new hash table for the updated matrix
  khash_t(entry) *new_hash_table = kh_init(entry);
  kh_resize(entry, new_hash_table, (mat->rows - 1) * mat->cols);

  // Iterate over the existing entries in the hash table
  for (khiter_t iter = kh_begin(mat->hash_table);
       iter != kh_end(mat->hash_table); ++iter) {
    if (kh_exist(mat->hash_table, iter)) {
      int64_t key = kh_key(mat->hash_table, iter);
      double value = kh_value(mat->hash_table, iter);
      size_t row = (size_t)(key >> 32);
      size_t col = (size_t)(key & 0xFFFFFFFF);

      if (row != row_idx) {
        if (row > row_idx) {
          row--;
        }

        int64_t new_key = ((int64_t)row << 32) | (int64_t)col;
        int ret = 0;
        khiter_t new_iter = kh_put(entry, new_hash_table, new_key, &ret);
        kh_value(new_hash_table, new_iter) = value;
      }
    }
  }

  // Free the old hash table
  kh_destroy(entry, mat->hash_table);

  // Update the hash table pointer in the matrix
  mat->hash_table = new_hash_table;
  mat->rows--;
}
