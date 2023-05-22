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

#include "dm_modify.h"
#include "dbg.h"
#include "dm.h"
#include "dm_internals.h"
#include "dm_io.h"
#include "dv_vector.h"

/*******************************/
/*    Retrieving Cols, Rows    */
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
/*   Set Cols, Rows    */
/*******************************/

void dm_set_column(DoubleMatrix *mat, size_t column_idx, DoubleVector *vec) {
  if (vec->rows != mat->rows) {
    perror("Error: Length of vector does not fit to number or matrix rows");
  } else {
    for (size_t i = 0; i < mat->rows; i++) {
      dm_set(mat, i, column_idx, dv_get(vec, i));
    }
  }
}

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
/*          Insert Row         */
/*******************************/

void dm_insert_row(DoubleMatrix *mat, size_t row_idx, DoubleVector *vec) {
  if (vec->rows != mat->cols) {
    perror("Error: Length of vector does not fit to number or matrix columns");
  } else {
    // resize the matrix:
    dm_resize(mat, mat->rows + 1, mat->cols);
    for (size_t i = 0; i < mat->cols; i++) {
      dm_set(mat, mat->rows - 1, i, dv_get(vec, i));
    }
  }
}

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
  case SPARSE:
    dm_resize_sparse(mat, new_row, new_col);
    break;
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
