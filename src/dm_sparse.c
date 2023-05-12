/**
 * @file dm_csr_matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 17-04-2023
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "dbg.h"
#include "dm_io.h"
#include "dm_math.h"
#include "dm_matrix.h"
#include <float.h>

enum { INIT_CAPACITY = 2U };
enum { RESIZE_FACTOR = 2 };

static bool is_zero(double value) { return fabs(value) < DBL_EPSILON; }

/*******************************/
/*        CSR Sparse Matrix    */
/*******************************/

DoubleMatrix *dm_create_sparse(size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  DoubleMatrix *mat = malloc(sizeof(DoubleMatrix));
  mat->rows = rows;
  mat->cols = cols;
  mat->capacity = INIT_CAPACITY;
  mat->nnz = 0;
  mat->row_indices =
      calloc(max_int(INIT_CAPACITY, (int)mat->nnz), sizeof(size_t));
  mat->col_indices =
      calloc(max_int(INIT_CAPACITY, (int)mat->nnz), sizeof(size_t));
  mat->format = SPARSE;
  mat->values = calloc(max_int(INIT_CAPACITY, (int)mat->nnz), sizeof(double));
  return mat;
}

/*******************************/
/*        Setter & Getter      */
/*******************************/

void dm_set_sparse(DoubleMatrix *mat, size_t i, size_t j, double value) {
  bool found = false;
  for (int k = 0; k < mat->nnz; k++) {
    if (mat->row_indices[k] == i && mat->col_indices[k] == j) {
      // if element is unequal zero, update the value
      if (is_zero(value) == false) {
        mat->values[k] = value;
        found = true;
      } else {
        found = true;
        dm_remove_zero(mat, i, j);
      }
    }
  }
  if (found == false) {
    dm_push_sparse(mat, i, j, value);
  }
}

// push new value to sparse matrix in COO format:
static void dm_push_sparse(DoubleMatrix *mat, size_t i, size_t j,
                           double value) {
  // check if value is zero:
  if (is_zero(value) == false) {
    // check if nnz is equal to capacity:
    if (mat->nnz == mat->capacity) {
      dm_realloc_sparse(mat, INIT_CAPACITY);
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
    if (mat->row_indices[k] == i && mat->col_indices[k] == j) {
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

double dm_get_sparse(const DoubleMatrix *mat, size_t i, size_t j) {
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
/*        Convert               */
/*******************************/

// convert dense matrix to sparse matrix of COO format:
void dm_convert_to_sparse(DoubleMatrix *mat) {
  // check if matrix is already in sparse format:
  if (mat->format == SPARSE) {
    printf("Matrix is already in sparse format!\n");
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
  size_t *row_indices = (size_t *)calloc(nnz, sizeof(size_t));
  size_t *col_indices = (size_t *)calloc(nnz, sizeof(size_t));
  double *values = (double *)calloc(nnz, sizeof(double));
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
/*        Resize               */
/*******************************/

static void dm_realloc_sparse(DoubleMatrix *mat, size_t new_capacity) {
  // check if matrix is already in sparse format:
  if (mat->format != SPARSE) {
    printf("Can not resize matrix to sparse format!\n");
    exit(EXIT_FAILURE);
  }

  // resize matrix:
  size_t *row_indices = (size_t *)realloc(
      mat->row_indices, (mat->capacity + new_capacity) * sizeof(size_t));
  size_t *col_indices = (size_t *)realloc(
      mat->col_indices, (mat->capacity + new_capacity) * sizeof(size_t));
  double *values = (double *)realloc(
      mat->values, (mat->capacity + new_capacity) * sizeof(double));
  if (row_indices == NULL || col_indices == NULL || values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  mat->capacity += new_capacity;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
}

// resize matrix of COO format:
void dm_resize_sparse(DoubleMatrix *mat, size_t new_row, size_t new_col) {
  // check if matrix is already in sparse format:
  if (mat->format != SPARSE) {
    printf("Can not resize matrix to sparse format!\n");
    exit(EXIT_FAILURE);
  }

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
