/**
 * @file dm_set.c
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

/*******************************/
/*          Set Value          */
/*******************************/

/**
 * @brief set value of index i, j
 *
 * @param mat
 * @param i,j
 * @param value
 */
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value) {
  switch (mat->format) {
  case SPARSE:
    dm_set_sparse(mat, i, j, value);
    break;
  case DENSE:
    dm_set_dense(mat, i, j, value);
    break;
  case HASHTABLE:
    dm_set_hash_table(mat, i, j, value);
    break;
  case VECTOR:
    dv_set(mat, i, value);
    break;
  }
}

/*******************************/
/*         Set DENSE           */
/*******************************/

// get value from dense matrix:
static void dm_set_dense(DoubleMatrix *mat, size_t i, size_t j,
                         const double value) {
  if (i < 0 || i >= mat->rows || j < 0 || j >= mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    exit(EXIT_FAILURE);
  }
  mat->values[i * mat->cols + j] = value;
  if (value != 0.0) {
    mat->nnz++;
  }
}

/*******************************/
/*         Set SPARSE          */
/*******************************/

static void dm_set_sparse(DoubleMatrix *mat, size_t i, size_t j, double value) {
  if (i < 0 || i >= mat->rows || j < 0 || j >= mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    exit(EXIT_FAILURE);
  }
  bool found = false;
  for (int k = 0; k < mat->nnz; k++) {
    if ((mat->row_indices[k] == i) && (mat->col_indices[k] == j)) {
      found = true;
      if (is_zero(value)) {
        dm_remove_zero(mat, i, j);
      } else {
        mat->values[k] = value;
      }
      break;
    }
  }
  if (found == false) {
    dm_push_sparse(mat, i, j, value);
  }
}

// push new value to sparse matrix in COO format:
static void dm_push_sparse(DoubleMatrix *mat, size_t i, size_t j,
                           double value) {
  if (i < 0 || i >= mat->rows || j < 0 || j >= mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    exit(EXIT_FAILURE);
  }
  // check if value is zero:
  if (is_zero(value) == false) {
    // check if nnz is equal to capacity:
    if (mat->nnz == mat->capacity) {
      dm_realloc_sparse(mat, mat->capacity * 2);
    }
    // push new value:
    mat->row_indices[mat->nnz] = i;
    mat->col_indices[mat->nnz] = j;
    mat->values[mat->nnz] = value;
    mat->nnz++;
  }
}

// remove zero value at index i,j of sparse matrix in COO format:
static void dm_remove_zero(DoubleMatrix *mat, size_t i, size_t j) {
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

/*******************************/
/*      Set HASHTABLE          */
/*******************************/

static void dm_set_hash_table(DoubleMatrix *matrix, size_t i, size_t j,
                              double value) {
  int ret = 0;
  khint_t k = 0;
  if (is_zero(value) == false) {
    // Calculate the key for the hash table using the combined row and column
    // indices
    int64_t key = (int64_t)i << 32 | (int64_t)j;

    // Check if the value already exists in the hash table
    k = kh_get(entry, matrix->hash_table, key);
    if (k != kh_end(matrix->hash_table)) {
      // Value already exists, update it
      kh_value(matrix->hash_table, k) = value;
    } else {
      // Value doesn't exist, insert it into the hash table
      k = kh_put(entry, matrix->hash_table, key, &ret);
      kh_value(matrix->hash_table, k) = value;
      matrix->nnz++;
    }
  }
}

/*******************************/
/*         Set VECTOR          */
/*******************************/

// see dv_vector.c
