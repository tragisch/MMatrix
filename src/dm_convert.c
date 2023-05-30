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
 * @brief convert matrix to format (COO, CSR, DENSE)
 * @param mat
 * @param format  (COO, CSR, DENSE)
 */
void dm_convert(DoubleMatrix *mat, matrix_format format) {
  if (mat->format == format) {
    return;
  }
  switch (format) {
  case DENSE:
    if (mat->format == COO) {
      dm_convert_coo_to_dense(mat);
    }
    break;
  case COO:
    if (mat->format == DENSE) {
      dm_convert_dense_to_coo(mat);
    }
    break;

  case VECTOR:
    break;
  }
}

/*******************************/
/*       DENSE -> COO       */
/*******************************/

// convert dense matrix to sparse matrix of COO format:
static void dm_convert_dense_to_coo(DoubleMatrix *mat) {
  // check if matrix is already in sparse format:
  if (mat->format == COO) {
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
  mat->format = COO;
  mat->nnz = nnz;
  mat->capacity = nnz;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
}

/*******************************/
/*       COO -> DENSE       */
/*******************************/

// convert SparseMatrix of COO format to Dense format
static void dm_convert_coo_to_dense(DoubleMatrix *mat) {
  if (mat->format == COO) {

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
