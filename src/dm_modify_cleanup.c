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

#include "dm.h"
#include "dm_modify.h"
#include <math.h>

#define EPSILON 1e-10

void dm_cleanup(DoubleMatrix *mat) {
  switch (mat->format) {
  case SPARSE:
    dm_cleanup_sparse(mat);
    break;
  case DENSE:
    dm_cleanup_dense(mat);
    break;
  case HASHTABLE:
    dm_cleanup_hashtable(mat);
    break;
  case VECTOR:
    break;
  }
}

static void dm_cleanup_sparse(DoubleMatrix *mat) {
  // remove all entries with absolute value < EPSILON:
  for (int i = 0; i < mat->nnz; i++) {
    if (fabs(mat->values[i]) < EPSILON) {
      dm_remove_entry(mat, mat->row_indices[i], mat->col_indices[i]);
      i--;
    }
  }
}

static void dm_cleanup_hashtable(DoubleMatrix *mat) {
  // remove all entries with absolute value < EPSILON:
  for (khiter_t iter = kh_begin(mat->hash_table);
       iter != kh_end(mat->hash_table); iter++) {
    if (kh_exist(mat->hash_table, iter)) {
      if (fabs(kh_value(mat->hash_table, iter)) < EPSILON) {
        kh_del(entry, mat->hash_table, iter);
      }
    }
  }
}

static void dm_cleanup_dense(DoubleMatrix *mat) {
  // remove all entries with absolute value < EPSILON:
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      if (fabs(dm_get(mat, i, j)) < EPSILON) {
        dm_set(mat, i, j, 0.0);
      }
    }
  }
}
