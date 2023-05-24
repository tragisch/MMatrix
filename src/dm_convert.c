/**
 * @file dm_convert.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm_convert.h"
#include "dm_math.h"
#include <stdio.h>

/*******************************/
/*  Convert format of matrix   */
/*******************************/

/**
 * @brief convert matrix to format (SPARSE, HASHTABLE, DENSE)
 * @param mat
 * @param format  (SPARSE, HASHTABLE, DENSE)
 */
void dm_convert(DoubleMatrix *mat, matrix_format format) {
  if (mat->format == format) {
    return;
  }
  switch (format) {
  case DENSE:
    if (mat->format == SPARSE) {
      dm_convert_sparse_to_dense(mat);
    } else if (mat->format == HASHTABLE) {
      dm_convert_hash_table_to_dense(mat);
    }
    break;
  case SPARSE:
    if (mat->format == DENSE) {
      dm_convert_dense_to_sparse(mat);
    } else if (mat->format == HASHTABLE) {
      dm_convert_hash_table_to_sparse(mat);
    }
    break;

  case HASHTABLE:
    if (mat->format == DENSE) {
      dm_convert_dense_to_hash_table(mat);
    } else if (mat->format == SPARSE) {
      dm_convert_sparse_to_hash_table(mat);
    }
    break;

  case VECTOR:
    break;
  }
}

/*******************************/
/*       DENSE -> SPARSE       */
/*******************************/

// convert dense matrix to sparse matrix of COO format:
static void dm_convert_dense_to_sparse(DoubleMatrix *mat) {
  // check if matrix is already in sparse format:
  if (mat->format == SPARSE) {
    printf("Matrix is already in sparse format!\n");
    return;
  }

  if (mat->format == VECTOR) {
    printf("Matrix is in vector format!\n");
    return;
  }

  // convert matrix:
  size_t nnz = 0;
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      if (is_zero(mat->values[i * mat->cols + j]) == false) {
        nnz++;
      }
    }
  }

  // allocate memory for sparse matrix:
  size_t *row_indices = (size_t *)calloc(nnz + 1, sizeof(size_t));
  size_t *col_indices = (size_t *)calloc(nnz + 1, sizeof(size_t));
  double *values = (double *)calloc(nnz + 1, sizeof(double));
  if (row_indices == NULL || col_indices == NULL || values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  // fill sparse matrix:
  size_t k = 0;
  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      if (is_zero(mat->values[i * mat->cols + j]) == false) {
        row_indices[k] = i;
        col_indices[k] = j;
        values[k] = mat->values[i * mat->cols + j];
        k++;
      }
    }
  }

  // free memory of dense matrix:
  free(mat->values);

  // set sparse matrix:
  mat->format = SPARSE;
  mat->nnz = nnz;
  mat->capacity = nnz;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
}

/*******************************/
/*       SPARSE -> DENSE       */
/*******************************/

// convert SparseMatrix of COO format to Dense format
static void dm_convert_sparse_to_dense(DoubleMatrix *mat) {
  if (mat->format == SPARSE) {

    // allocate memory for dense matrix:
    double *new_values =
        (double *)calloc(mat->rows * mat->cols, sizeof(double));
    size_t new_capacity = mat->rows * mat->cols;

    // fill dense matrix with values from sparse matrix:
    for (int i = 0; i < mat->nnz; i++) {
      new_values[mat->row_indices[i] * mat->cols + mat->col_indices[i]] =
          mat->values[i];
    }

    mat->format = DENSE;
    mat->values = new_values;
    mat->capacity = new_capacity;

    // free memory of sparse matrix:
    free(mat->row_indices);
    free(mat->col_indices);
    mat->row_indices = NULL;
    mat->col_indices = NULL;
  }
}

/*******************************/
/*     SPARSE -> HASHTABLE     */
/*******************************/

static void dm_convert_sparse_to_hash_table(DoubleMatrix *mat) {
  if (mat->format == SPARSE) {
    // Create hash table
    mat->hash_table = kh_init(entry);
    if (mat->hash_table == NULL) {
      printf("Error allocating memory!\n");
      exit(EXIT_FAILURE);
    }

    // fill hash table:
    khint_t k = 0;
    int ret = 0;
    for (int i = 0; i < mat->nnz; i++) {
      int64_t key =
          (int64_t)mat->row_indices[i] << 32 | (int64_t)mat->col_indices[i];
      // Check if the value already exists in the hash table
      k = kh_get(entry, mat->hash_table, key);
      if (k != kh_end(mat->hash_table)) {
        // Value already exists, update it
        kh_value(mat->hash_table, k) = mat->values[i];
      } else {
        // Value doesn't exist, insert it into the hash table
        k = kh_put(entry, mat->hash_table, key, &ret);
        kh_value(mat->hash_table, k) = mat->values[i];
      }
    }

    // free memory of sparse matrix:
    free(mat->row_indices);
    free(mat->col_indices);
    free(mat->values);

    // set hash_table matrix:
    mat->format = HASHTABLE;
    mat->row_indices = NULL;
    mat->col_indices = NULL;
    mat->values = NULL;
    mat->capacity = 0;
  }
}

/*******************************/
/*     DENSE  -> HASHTABLE     */
/*******************************/

// convert dense to hast_table:
static void dm_convert_dense_to_hash_table(DoubleMatrix *mat) {
  if (mat->format == DENSE) {
    // Create hash table
    mat->hash_table = kh_init(entry);
    if (mat->hash_table == NULL) {
      printf("Error allocating memory!\n");
      exit(EXIT_FAILURE);
    }

    // fill hash table:
    khint_t k = 0;
    int ret = 0;
    for (int i = 0; i < mat->rows; i++) {
      for (int j = 0; j < mat->cols; j++) {
        double value = dm_get(mat, i, j);
        if (value != 0) {
          int64_t key = (int64_t)i << 32 | j;
          // Check if the value already exists in the hash table
          k = kh_get(entry, mat->hash_table, key);
          if (k != kh_end(mat->hash_table)) {
            // Value already exists, update it
            kh_value(mat->hash_table, k) = value;
          } else {
            // Value doesn't exist, insert it into the hash table
            k = kh_put(entry, mat->hash_table, key, &ret);
            kh_value(mat->hash_table, k) = value;
            mat->nnz++;
          }
        }
      }
    }

    // free memory of dense matrix:
    free(mat->values);

    // set hash_table matrix:
    mat->format = HASHTABLE;
    mat->values = NULL;
    mat->capacity = 0;
  }
}

/*******************************/
/*     HASHTABLE --> SPARSE    */
/*******************************/

// convert hash_table matrix to COO format
static void dm_convert_hash_table_to_sparse(DoubleMatrix *mat) {
  if (mat->format == HASHTABLE) {

    // allocate memory for COO matrix:
    size_t *row_indices = (size_t *)calloc(mat->nnz, sizeof(size_t));
    size_t *col_indices = (size_t *)calloc(mat->nnz, sizeof(size_t));
    size_t nnz = 0;
    double *values = (double *)calloc(mat->nnz, sizeof(double));
    if (row_indices == NULL || col_indices == NULL || values == NULL) {
      printf("Error allocating memory!\n");
      exit(EXIT_FAILURE);
    }

    // fill COO matrix:
    size_t k = 0;
    for (int i = 0; i < mat->rows; i++) {
      for (int j = 0; j < mat->cols; j++) {
        double value = dm_get(mat, i, j);
        if (value != 0) {
          row_indices[k] = i;
          col_indices[k] = j;
          values[k] = value;
          nnz++;
        }
      }
    }

    // free memory of hash_table matrix:
    kh_destroy(entry, mat->hash_table);

    // set COO matrix:
    mat->format = SPARSE;
    mat->row_indices = row_indices;
    mat->col_indices = col_indices;
    mat->values = values;
    mat->nnz = nnz;
    mat->capacity = nnz;
  }
}

/*******************************/
/*     HASHTABLE --> DENSE     */
/*******************************/

static void dm_convert_hash_table_to_dense(DoubleMatrix *mat) {
  if (mat->format == HASHTABLE) {
    // allocate memory for dense matrix:
    double *values = (double *)calloc(mat->rows * mat->cols, sizeof(double));
    if (values == NULL) {
      printf("Error allocating memory!\n");
      exit(EXIT_FAILURE);
    }

    // fill dense matrix:
    khint_t k = 0;
    for (int i = 0; i < mat->rows; i++) {
      for (int j = 0; j < mat->cols; j++) {
        int64_t key = (int64_t)i << 32 | j;
        // Check if the value already exists in the hash table
        k = kh_get(entry, mat->hash_table, key);
        if (k != kh_end(mat->hash_table)) {
          // Value already exists, update it
          values[i * mat->cols + j] = kh_value(mat->hash_table, k);
        }
      }
    }

    // free memory of hash_table matrix:
    kh_destroy(entry, mat->hash_table);

    // set dense matrix:
    mat->format = DENSE;
    mat->values = values;
    mat->capacity = mat->rows * mat->cols;
  }
}
