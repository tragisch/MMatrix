/**
 * @file sp_matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 17-04-2023
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "sp_matrix.h"
#include "dm_matrix.h"

// debug:
#include "dbg.h"
#include <assert.h>

// #define NDEBUG
enum { INIT_CAPACITY = 2U };

/*******************************/
/*        Sparse Matrix        */
/*******************************/

SparseMatrix *sp_create(size_t rows, size_t cols) {
  SparseMatrix *sp_matrix = malloc(sizeof(SparseMatrix));
  sp_matrix->rows = rows;
  sp_matrix->cols = cols;
  sp_matrix->nnz = 0;
  sp_matrix->is_sparse = true;
  sp_matrix->row_indices = NULL;
  sp_matrix->col_indices = NULL;
  sp_matrix->values = NULL;
  sp_matrix->is_sparse = true;
  return sp_matrix;
}

SparseMatrix *sp_create_rand(size_t rows, size_t cols, double density) {
  SparseMatrix *mat = malloc(sizeof(SparseMatrix));
  mat->rows = rows;
  mat->cols = cols;
  mat->nnz = 0;
  mat->values = malloc(0);
  mat->row_indices = malloc(0);
  mat->col_indices = malloc(0);
  mat->is_sparse = true;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (randomDouble() < density) {
        double value = randomDouble();
        mat->values = realloc(mat->values, (mat->nnz + 1) * sizeof(double));
        mat->row_indices =
            realloc(mat->row_indices, (mat->nnz + 1) * sizeof(int));
        mat->col_indices =
            realloc(mat->col_indices, (mat->nnz + 1) * sizeof(int));
        mat->values[mat->nnz] = value;
        mat->row_indices[mat->nnz] = i;
        mat->col_indices[mat->nnz] = j;
        mat->nnz++;
      }
    }
  }

  return mat;
}

// setup directly COO format for sparse matrix:
SparseMatrix *sp_create_from_array(size_t rows, size_t cols, size_t nnz,
                                   size_t *row_indices, size_t *col_indices,
                                   double *values) {
  SparseMatrix *sp_matrix = sp_create(rows, cols);
  sp_matrix->nnz = nnz;
  sp_matrix->col_indices = col_indices;
  sp_matrix->values = values;
  sp_matrix->row_indices = row_indices;
  sp_matrix->is_sparse = true;
  return sp_matrix;
}

// free sparse matrix
void sp_destroy(SparseMatrix *sp_matrix) {
  if (sp_matrix == NULL) {
    return;
  }
  if (sp_matrix->col_indices != NULL) {
    free(sp_matrix->col_indices);
    sp_matrix->col_indices = NULL;
  }
  if (sp_matrix->values != NULL) {
    free(sp_matrix->values);
    sp_matrix->values = NULL;
  }
  if (sp_matrix->row_indices != NULL) {
    free(sp_matrix->row_indices);
    sp_matrix->row_indices = NULL;
  }
  free(sp_matrix);
}

double sp_get(const SparseMatrix *mat, size_t i, size_t j) {
  if (mat->is_sparse == false) {
    return dm_get(mat, i, j);
  }
  // search for the element with row i and column j
  for (int k = 0; k < mat->nnz; k++) {
    if (mat->row_indices[k] == i && mat->col_indices[k] == j) {
      return mat->values[k];
    }
  }
  // return 0.0 if element is not found
  return 0.0;
}

void sp_set(SparseMatrix *mat, size_t i, size_t j, double value) {
  if (mat->is_sparse == false) {
    dm_set(mat, i, j, value);
    return;
  }
  // search for the element with row i and column j
  for (int k = 0; k < mat->nnz; k++) {
    if (mat->row_indices[k] == i && mat->col_indices[k] == j) {
      // update the value if element is found
      mat->values[k] = value;
      return;
    }
  }
  // if element is not found, add it to the matrix
  mat->values = realloc(mat->values, (mat->nnz + 1) * sizeof(double));
  mat->row_indices = realloc(mat->row_indices, (mat->nnz + 1) * sizeof(int));
  mat->col_indices = realloc(mat->col_indices, (mat->nnz + 1) * sizeof(int));
  mat->values[mat->nnz] = value;
  mat->row_indices[mat->nnz] = i;
  mat->col_indices[mat->nnz] = j;
  mat->nnz++;
}

/*
 * @brief get density of sparse matrix
 *
 * @param mat
 */

double sp_density(const SparseMatrix *mat) {
  return (double)mat->nnz / (mat->rows * mat->cols);
}

/*
 * @brief resize sparse matrix
 *
 * @param mat
 */
void sp_resize(SparseMatrix *mat, size_t new_rows, size_t new_cols) {
  if (mat == NULL) {
    perror("Error: null pointer passed to sp_resize.\n");
    return;
  }
  if (new_rows < mat->rows || new_cols < mat->cols) {
    perror("Error: cannot shrink matrix.\n");
    return;
  }

  mat->values = realloc(mat->values, sizeof(double) * (mat->nnz));
  mat->row_indices = realloc(mat->row_indices, sizeof(int) * (mat->nnz));
  mat->col_indices = realloc(mat->col_indices, sizeof(int) * (mat->nnz));

  for (int i = mat->rows; i < new_rows; i++) {
    for (int j = 0; j < new_cols; j++) {
      mat->values = realloc(mat->values, sizeof(double) * (mat->nnz + 1));
      mat->row_indices =
          realloc(mat->row_indices, sizeof(int) * (mat->nnz + 1));
      mat->col_indices =
          realloc(mat->col_indices, sizeof(int) * (mat->nnz + 1));
      mat->values[mat->nnz] = 0.0;
      mat->row_indices[mat->nnz] = i;
      mat->col_indices[mat->nnz] = j;
      mat->nnz++;
    }
  }

  mat->rows = new_rows;
  mat->cols = new_cols;
}

// return true if all fields of SparseMatrix a inialized correctly
bool sp_is_valid(const SparseMatrix *a) {
  if (a == NULL) {
    return false;
  }
  if (a->rows == 0 || a->cols == 0) {
    return false;
  }
  if (a->nnz == 0) {
    return false;
  }
  if (a->row_indices == NULL) {
    return false;
  }
  if (a->col_indices == NULL) {
    return false;
  }
  if (a->values == NULL) {
    return false;
  }
  return true;
}

// convert a DoubleMatriy (dense) to a SparseMatrix
SparseMatrix *sp_convert_to_sparse(const DoubleMatrix *dense) {
  SparseMatrix *sparse = sp_create(dense->rows, dense->cols);

  // Count the number of non-zero elements
  for (size_t i = 0; i < dense->rows; i++) {
    for (size_t j = 0; j < dense->cols; j++) {
      if (dense->values[i * dense->cols + j] != 0.0) {
        sparse->nnz++;
      }
    }
  }

  // Allocate memory for the COO format
  sparse->row_indices = (size_t *)malloc((sparse->nnz + 1) * sizeof(size_t));
  sparse->col_indices = (size_t *)malloc(sparse->nnz * sizeof(size_t));
  sparse->values = (double *)malloc(sparse->nnz * sizeof(double));

  // Fill in the COO format
  size_t k = 0;
  for (size_t i = 0; i < dense->rows; i++) {
    for (size_t j = 0; j < dense->cols; j++) {
      double val = dense->values[i * dense->cols + j];
      if (val != 0.0) {
        sparse->col_indices[k] = j;
        sparse->row_indices[k] = i;
        sparse->values[k] = val;
        k++;
      }
    }
  }
  return sparse;
}
