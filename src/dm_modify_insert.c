/**
 * @file dm_modify_getset.c
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
#include "dm_modify.h"
#include "dv_vector.h"

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
    case SPARSE:
      dm_insert_column_sparse(mat, column_idx);
      break;
    case HASHTABLE:
      dm_insert_column_hashtable(mat, column_idx);
      break;
    case VECTOR:
      break;
    }

    // insert the new column:
    for (size_t i = 0; i < mat->rows; i++) {
      dm_set(mat, i, column_idx, dv_get(vec, i));
    }
  }
}

static void dm_insert_column_sparse(DoubleMatrix *mat, size_t column_idx) {
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

static void dm_insert_column_hashtable(DoubleMatrix *mat, size_t column_idx) {
  // resize hast_table
  mat->cols++;
  kh_resize(entry, mat->hash_table, mat->rows * mat->cols);

  // Create a new hash table for the updated matrix
  khash_t(entry) *new_hash_table = kh_init(entry);

  // Iterate over the existing entries in the hash table
  for (khiter_t iter = kh_begin(matrix->hash_table);
       iter != kh_end(mat->hash_table); ++iter) {
    if (kh_exist(mat->hash_table, iter)) {
      int64_t key = kh_key(mat->hash_table, iter);
      int64_t new_key = key + ((key >> 32) >= column_idx ? 1 : 0);
      double value = kh_value(mat->hash_table, iter);

      // Insert the updated entry into the new hash table
      int ret = 0;
      iter = kh_put(entry, new_hash_table, new_key, &ret);
      kh_value(new_hash_table, iter) = value;
    }
  }

  // Free the old hash table
  kh_destroy(entry, mat->hash_table);

  // Update the hash table pointer in the matrix
  mat->hash_table = new_hash_table;
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
    case SPARSE:
      dm_insert_row_sparse(mat, row_idx);
      break;
    case HASHTABLE:
      dm_insert_row_hashtable(mat, row_idx);
      break;
    case VECTOR:
      break;
    }

    // insert the new row:
    for (size_t i = 0; i < mat->cols; i++) {
      dm_set(mat, row_idx, i, dv_get(vec, i));
    }
  }
}

static void dm_insert_row_sparse(DoubleMatrix *mat, size_t row_idx) {
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

static void dm_insert_row_hashtable(DoubleMatrix *mat, size_t row_idx) {
  // Resize the hash_table
  mat->rows++;
  kh_resize(entry, mat->hash_table, mat->rows * mat->cols);

  // Create a new hash table for the updated matrix
  khash_t(entry) *new_hash_table = kh_init(entry);

  // Iterate over the existing entries in the hash table
  for (khiter_t iter = kh_begin(mat->hash_table);
       iter != kh_end(mat->hash_table); ++iter) {
    if (kh_exist(mat->hash_table, iter)) {
      int64_t key = kh_key(mat->hash_table, iter);
      size_t row = (size_t)(key >> 32);
      size_t col = (size_t)(key & 0xFFFFFFFF);

      if (row >= row_idx) {
        key = ((int64_t)(row + 1) << 32) | (int64_t)col;
      }

      double value = kh_value(mat->hash_table, iter);

      // Insert the updated entry into the new hash table
      int ret = 0;
      iter = kh_put(entry, new_hash_table, key, &ret);
      kh_value(new_hash_table, iter) = value;
    }
  }

  // Free the old hash table
  kh_destroy(entry, mat->hash_table);

  // Update the hash table pointer in the matrix
  mat->hash_table = new_hash_table;
}
