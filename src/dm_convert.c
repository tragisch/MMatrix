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
 * @brief convert matrix to format (COO, CSC, DENSE)
 * @param mat
 * @param format  (COO, CSC, DENSE)
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
    if (mat->format == CSC) {
      dm_convert_csc_to_dense(mat);
    }
    break;
  case COO:
    if (mat->format == DENSE) {
      dm_convert_dense_to_coo(mat);
    } else if (mat->format == CSC) {
      dm_convert_csc_to_coo(mat);
    }
    break;
  case CSC:
    if (mat->format == DENSE) {
      dm_convert_dense_to_csc(mat);
    } else if (mat->format == COO) {
      dm_convert_coo_to_csc(mat);
    }
    break;
  case VECTOR:
    break;
  }
}

/*******************************/
/*       DENSE -> COO          */
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
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
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
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
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
/*       DENSE -> CSC          */
/*******************************/

// convert dense matrix to sparse matrix of CSC format:
static void dm_convert_dense_to_csc(DoubleMatrix *mat) {
  // check if matrix is already in sparse format:
  if (mat->format == CSC) {
    printf("Matrix is already in sparse format!\n");
    return;
  }

  if (mat->format == VECTOR) {
    printf("Matrix is in vector format!\n");
    return;
  }

  // convert matrix:
  size_t nnz = 0;
  for (size_t j = 0; j < mat->cols; j++) {
    for (size_t i = 0; i < mat->rows; i++) {
      if (is_zero(mat->values[i * mat->cols + j]) == false) {
        nnz++;
      }
    }
  }

  // allocate memory for sparse matrix:
  size_t *row_indices = (size_t *)calloc(mat->cols + 1, sizeof(size_t));
  size_t *col_indices = (size_t *)calloc(nnz + 1, sizeof(size_t));
  double *values = (double *)calloc(nnz + 1, sizeof(double));
  if (row_indices == NULL || col_indices == NULL || values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  // fill sparse matrix:
  size_t k = 0;
  for (size_t j = 0; j < mat->cols; j++) {
    for (size_t i = 0; i < mat->rows; i++) {
      if (is_zero(mat->values[i * mat->cols + j]) == false) {
        col_indices[k] = j;
        values[k] = mat->values[i * mat->cols + j];
        k++;
      }
    }
    row_indices[j + 1] = k;
  }

  // free memory of dense matrix:
  free(mat->values);

  // set sparse matrix:
  mat->format = CSC;
  mat->nnz = nnz;
  mat->capacity = nnz;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
}

/*******************************/
/*         COO -> CSC          */
/*******************************/

// convert SparseMatrix to CSC format:
static void dm_convert_coo_to_csc(DoubleMatrix *mat) {
  if (mat->format == CSC) {
    // Already in CSC format, nothing to do
    return;
  }
  if (mat->format == COO) {
    // Allocate memory for new arrays
    size_t *new_row_indices = calloc(mat->cols + 1, sizeof(size_t));
    size_t *new_col_indices = calloc(mat->nnz, sizeof(size_t));
    double *new_values = calloc(mat->nnz, sizeof(double));

    // Compute number of non-zero elements in each column
    for (size_t i = 0; i < mat->nnz; i++) {
      new_row_indices[mat->col_indices[i] + 1]++;
    }
    for (size_t j = 1; j <= mat->cols; j++) {
      new_row_indices[j] += new_row_indices[j - 1];
    }

    // Copy values to new arrays
    for (size_t i = 0; i < mat->rows; i++) {
      for (size_t k = mat->row_indices[i]; k < mat->row_indices[i + 1]; k++) {
        size_t j = mat->col_indices[k];
        size_t l = new_row_indices[j];

        new_col_indices[l] = i;
        new_values[l] = mat->values[k];

        new_row_indices[j]++;
      }
    }

    // Shift column pointers back
    for (size_t j = mat->cols; j > 0; j--) {
      new_row_indices[j] = new_row_indices[j - 1];
    }
    new_row_indices[0] = 0;

    // Update matrix struct
    free(mat->row_indices);
    free(mat->col_indices);
    free(mat->values);
    mat->row_indices = new_row_indices;
    mat->col_indices = new_col_indices;
    mat->values = new_values;
    mat->format = CSC;
  } else {
    // Unsupported format
    fprintf(stderr, "Error: Unsupported format\n");
    exit(EXIT_FAILURE);
  }
}

/*******************************/
/*         COO -> DENSE        */
/*******************************/

// convert COO to dense format:
static void dm_convert_coo_to_dense(DoubleMatrix *mat) {
  // check if matrix is already in dense format:
  if (mat->format == DENSE) {
    printf("Matrix is already in dense format!\n");
    return;
  }

  // convert matrix:
  double *values = (double *)calloc(mat->rows * mat->cols, sizeof(double));
  if (values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      values[i * mat->cols + j] = 0.0;
    }
  }

  for (size_t k = 0; k < mat->nnz; k++) {
    values[mat->row_indices[k] * mat->cols + mat->col_indices[k]] =
        mat->values[k];
  }

  // free memory of sparse matrix:
  free(mat->row_indices);
  free(mat->col_indices);
  free(mat->values);

  // set dense matrix:
  mat->format = DENSE;
  mat->nnz = mat->rows * mat->cols;
  mat->capacity = mat->rows * mat->cols;
  mat->row_indices = NULL;
  mat->col_indices = NULL;
  mat->values = values;
}

/*******************************/
/*         CSC -> DENSE        */
/*******************************/

// convert CSC to dense format:
static void dm_convert_csc_to_dense(DoubleMatrix *mat) {
  // check if matrix is already in dense format:
  if (mat->format == DENSE) {
    printf("Matrix is already in dense format!\n");
    return;
  }

  // convert matrix:
  double *values = (double *)calloc(mat->rows * mat->cols, sizeof(double));
  if (values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      values[i * mat->cols + j] = 0.0;
    }
  }

  for (size_t j = 0; j < mat->cols; j++) {
    for (size_t k = mat->row_indices[j]; k < mat->row_indices[j + 1]; k++) {
      values[mat->col_indices[k] * mat->cols + j] = mat->values[k];
    }
  }

  // free memory of sparse matrix:
  free(mat->row_indices);
  free(mat->col_indices);
  free(mat->values);

  // set dense matrix:
  mat->format = DENSE;
  mat->nnz = mat->rows * mat->cols;
  mat->capacity = mat->rows * mat->cols;
  mat->row_indices = NULL;
  mat->col_indices = NULL;
  mat->values = values;
}

/*******************************/
/*         CSC -> COO          */
/*******************************/

// convert CSC to COO format:
static void dm_convert_csc_to_coo(DoubleMatrix *mat) {
  // check if matrix is already in COO format:
  if (mat->format == COO) {
    printf("Matrix is already in COO format!\n");
    return;
  }

  // convert matrix:
  size_t *row_indices = (size_t *)calloc(mat->nnz, sizeof(size_t));
  size_t *col_indices = (size_t *)calloc(mat->nnz, sizeof(size_t));
  double *values = (double *)calloc(mat->nnz, sizeof(double));
  if (row_indices == NULL || col_indices == NULL || values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  size_t k = 0;
  for (size_t j = 0; j < mat->cols; j++) {
    for (size_t i = mat->row_indices[j]; i < mat->row_indices[j + 1]; i++) {
      row_indices[k] = mat->col_indices[i];
      col_indices[k] = j;
      values[k] = mat->values[i];
      k++;
    }
  }

  // free memory of sparse matrix:
  free(mat->row_indices);
  free(mat->col_indices);
  free(mat->values);

  // set COO matrix:
  mat->format = COO;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
}