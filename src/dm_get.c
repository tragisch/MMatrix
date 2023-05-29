/**
 * @file dm_get.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm.h"
#include "dm_internals.h"
#include "dm_math.h"
#include "dm_vector.h"

/*******************************/
/*          Get Value          */
/*******************************/

/**
 * @brief get value of index i, j
 *
 * @param mat
 * @param i,j
 * @return double
 */
double dm_get(const DoubleMatrix *mat, size_t i, size_t j) {
  // perror if boundaries are exceeded
  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    exit(EXIT_FAILURE);
  }
  switch (mat->format) {
  case DENSE:
    return dm_get_dense(mat, i, j);
    break;
  case SPARSE:
    return dm_get_sparse(mat, i, j);
    break;
  case HASHTABLE:
    return dm_get_hash_table(mat, i, j);
    break;
  case VECTOR:
    return dv_get(mat, i);
    break;
  }
}

/*******************************/
/*         Get DENSE           */
/*******************************/

// get value from dense matrix:
static double dm_get_dense(const DoubleMatrix *mat, size_t i, size_t j) {
  return mat->values[i * mat->cols + j];
}

/*******************************/
/*         Get SPARSE          */
/*******************************/

// get value of index i, j of sparse matrix in COO format:
static double dm_get_sparse(const DoubleMatrix *mat, size_t i, size_t j) {
  for (int k = 0; k < mat->nnz; k++) {
    if (mat->row_indices[k] == i && mat->col_indices[k] == j) {
      // Element found, return value
      return mat->values[k];
    }
  }

  // Element not found, return 0.0
  return 0.0;
}

/*******************************/
/*         Get HASHTABLE       */
/*******************************/

static double dm_get_hash_table(const DoubleMatrix *matrix, size_t i,
                                size_t j) {
  // Calculate the key for the hash table using the combined row and column
  // indices
  int64_t key = (int64_t)i << 32 | (int64_t)j;

  // Check if the value exists in the hash table
  khint_t k = kh_get(entry, matrix->hash_table, key);
  if (k != kh_end(matrix->hash_table)) {
    // Value exists, return it
    return kh_value(matrix->hash_table, k);
  } // Value doesn't exist, return 0.0
  return 0.0;
}

/*******************************/
/*         Get VECTOR          */
/*******************************/

// see dv_vector.c