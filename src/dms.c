/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "dms.h"

#include <omp.h>
#include <pcg_variants.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#ifndef INIT_CAPACITY
#define INIT_CAPACITY 100
#endif
#ifndef EPSILON
#define EPSILON 1e-9
#endif

/*******************************/
/*       Private Functions     */
/*******************************/

double dms_max_double(double a, double b) { return a > b ? a : b; }
double dms_min_double(double a, double b) { return a < b ? a : b; }
int dms_max_int(int a, int b) { return a > b ? a : b; }

static size_t __dms_binary_search(const DoubleSparseMatrix *matrix, size_t i,
                                  size_t j) {
  size_t low = 0;
  size_t high = matrix->nnz;

  while (low < high) {
    size_t mid = (low + high) / 2;

    if (matrix->row_indices[mid] == i && matrix->col_indices[mid] == j) {
      return mid; // Element found at position (i, j)
    }
    if (matrix->row_indices[mid] < i ||
        (matrix->row_indices[mid] == i && matrix->col_indices[mid] < j)) {
      low = mid + 1; // Search in the upper half
    } else {
      high = mid; // Search in the lower half
    }
  }

  return low; // Element not found, return the insertion position
}

static void __dms_insert_element(DoubleSparseMatrix *matrix, size_t i, size_t j,
                                 double value, size_t position) {
  // Increase the capacity if needed
  if (matrix->nnz == matrix->capacity) {
    dms_realloc(matrix, matrix->capacity * 2);
  }

  // Shift the existing elements to make space for the new element
  for (size_t k = matrix->nnz; k > position; k--) {
    matrix->row_indices[k] = matrix->row_indices[k - 1];
    matrix->col_indices[k] = matrix->col_indices[k - 1];
    matrix->values[k] = matrix->values[k - 1];
  }

  // Insert the new element at the appropriate position
  matrix->row_indices[position] = i;
  matrix->col_indices[position] = j;
  matrix->values[position] = value;

  // Increment the count of non-zero elements
  matrix->nnz++;
}

/*******************************/
/*       Public Functions     */
/*******************************/

DoubleSparseMatrix *dms_create_with_values(size_t rows, size_t cols, size_t nnz,
                                           size_t *row_indices,
                                           size_t *col_indices,
                                           double *values) {
  DoubleSparseMatrix *mat = dms_create(rows, cols, nnz);
  for (size_t i = 0; i < nnz; i++) {
    mat->row_indices[i] = row_indices[i];
    mat->col_indices[i] = col_indices[i];
    mat->values[i] = values[i];
  }
  mat->nnz = nnz;
  return mat;
}

DoubleSparseMatrix *dms_create_empty(void) {
  DoubleSparseMatrix *mat = malloc(sizeof(DoubleSparseMatrix));
  if (!mat) {
    perror("Error allocating memory for matrix struct");
    return NULL;
  }

  mat->rows = 0;
  mat->cols = 0;
  mat->nnz = 0;
  mat->capacity = 0;
  mat->row_indices = NULL;
  mat->col_indices = NULL;
  mat->values = NULL;

  return mat;
}

DoubleSparseMatrix *dms_create(size_t rows, size_t cols, size_t capacity) {
  if (rows < 1 || cols < 1) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }

  DoubleSparseMatrix *mat = malloc(sizeof(DoubleSparseMatrix));
  if (!mat) {
    perror("Error allocating memory for matrix struct");
    return NULL;
  }

  mat->rows = rows;
  mat->cols = cols;
  mat->nnz = 0;
  mat->capacity = capacity;

  if (mat->capacity == 0) {
    perror("Error: matrix capacity cannot be zero.");
    free(mat);
    return NULL;
  }

  mat->row_indices = calloc(mat->capacity, sizeof(size_t));
  if (!mat->row_indices) {
    perror("Error allocating memory for row indices");
    free(mat);
    return NULL;
  }

  mat->col_indices = calloc(mat->capacity, sizeof(size_t));
  if (!mat->col_indices) {
    perror("Error allocating memory for column indices");
    free(mat->row_indices);
    free(mat);
    return NULL;
  }

  mat->values = calloc(mat->capacity, sizeof(double));
  if (!mat->values) {
    perror("Error allocating memory for values");
    free(mat->col_indices);
    free(mat->row_indices);
    free(mat);
    return NULL;
  }

  return mat;
}

DoubleSparseMatrix *dms_create_clone(const DoubleSparseMatrix *m) {
  DoubleSparseMatrix *copy = dms_create_empty();
  copy->rows = m->rows;
  copy->cols = m->cols;
  copy->nnz = m->nnz;
  copy->capacity = m->capacity;

  copy->row_indices = calloc(m->capacity, sizeof(size_t));
  copy->col_indices = calloc(m->capacity, sizeof(size_t));
  copy->values = calloc(m->capacity, sizeof(double));
  for (size_t i = 0; i < m->nnz; i++) {
    copy->row_indices[i] = m->row_indices[i];
    copy->col_indices[i] = m->col_indices[i];
    copy->values[i] = m->values[i];
  }
  return copy;
}

DoubleSparseMatrix *dms_create_identity(size_t n) {
  if (n < 1) {
    perror("Error: invalid identity dimensions.\n");
    return NULL;
  }
  DoubleSparseMatrix *mat = dms_create(n, n, n + 1);
  for (size_t i = 0; i < n; i++) {
    mat->row_indices[i] = i;
    mat->col_indices[i] = i;
    mat->values[i] = 1.0;
  }
  mat->nnz = n;
  return mat;
}

cs *dms_to_cs(const DoubleSparseMatrix *coo) {
  int m = (int)coo->rows;
  int n = (int)coo->cols;
  int nz = (int)coo->nnz;

  // Allocate a CSparse matrix in COO format
  cs *T = cs_spalloc(m, n, nz, 1, 1);
  if (!T)
    return NULL;

  // Fill the CSparse matrix with the data from the DoubleSparseMatrix
  for (size_t k = 0; k < (size_t)nz; k++) {
    cs_entry(T, (int32_t)coo->row_indices[k], (int32_t)coo->col_indices[k], coo->values[k]);
  }

  // Convert the COO matrix to CSC format
  cs *A = cs_compress(T);
  cs_spfree(T); // Free the temporary COO matrix

  return A;
}

DoubleSparseMatrix *cs_to_dms(const cs *A) {
  // Allocate memory for the DoubleSparseMatrix structure
  DoubleSparseMatrix *coo =
      (DoubleSparseMatrix *)malloc(sizeof(DoubleSparseMatrix));
  if (!coo)
    return NULL;

  coo->rows = (size_t)A->m;
  coo->cols = (size_t)A->n;
  coo->nnz = (size_t)A->nzmax;
  coo->capacity = (size_t)(A->nzmax + INIT_CAPACITY);

  // Allocate memory for the COO arrays
  coo->row_indices =
      (size_t *)malloc((size_t)dms_max_int((int)coo->nnz, (int)coo->capacity) * sizeof(size_t));
  coo->col_indices =
      (size_t *)malloc((size_t)dms_max_int((int)coo->nnz, (int)coo->capacity) * sizeof(size_t));
  coo->values =
      (double *)malloc((size_t)dms_max_int((int)coo->nnz, (int)coo->capacity) * sizeof(double));

  if (!coo->row_indices || !coo->col_indices || !coo->values) {
    free(coo->row_indices);
    free(coo->col_indices);
    free(coo->values);
    free(coo);
    return NULL;
  }

  // Fill the COO arrays with the data from the CSC matrix
  size_t nnz_index = 0;
  for (int col = 0; col < A->n; col++) {
    for (int p = A->p[col]; p < A->p[col + 1]; p++) {
      coo->row_indices[nnz_index] = (size_t)A->i[p];
      coo->col_indices[nnz_index] = (size_t)col;
      coo->values[nnz_index] = A->x[p];
      nnz_index++;
    }
  }

  return coo;
}

DoubleSparseMatrix *dms_create_random(size_t rows, size_t cols,
                                      double density) {
  double nnz_d = (double)rows * (double)cols * density;
  size_t nnz = (size_t)nnz_d;
  if (nnz == 0)
    return dms_create(rows, cols, 0);

  DoubleSparseMatrix *mat = dms_create(rows, cols, nnz);
  if (!mat)
    return NULL;

  unsigned int global_seed =
      (unsigned int)((uintptr_t)mat ^ (uintptr_t)time(NULL));

#pragma omp parallel
  {
    pcg32_random_t rng;
    int thread_id = omp_get_thread_num();
    pcg32_srandom_r(&rng, global_seed ^ (unsigned int)thread_id, (unsigned int)thread_id);

  #pragma omp for 
    for (size_t k = 0; k < nnz; ++k) {
      mat->row_indices[k] = pcg32_random_r(&rng) % rows;
      mat->col_indices[k] = pcg32_random_r(&rng) % cols;
      mat->values[k] = (double)pcg32_random_r(&rng) / UINT32_MAX;
    }
  }

  mat->nnz = nnz;
  return mat;
}

DoubleSparseMatrix *dms_create_from_array(size_t rows, size_t cols,
                                          double *array) {
  DoubleSparseMatrix *mat = dms_create(rows, cols, rows * cols);
  size_t k = 0;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (array[i * cols + j] != 0) {
        mat->row_indices[k] = i;
        mat->col_indices[k] = j;
        mat->values[k] = array[i * cols + j];
        k++;
      }
    }
  }
  mat->nnz = k;
  return mat;
}

DoubleSparseMatrix *dms_create_from_2D_array(size_t rows, size_t cols,
                                             double array[rows][cols]) {
  size_t nnz = 0;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (array[i][j] != 0) {
        nnz++;
      }
    }
  }
  DoubleSparseMatrix *mat = dms_create(rows, cols, nnz + 1);
  size_t k = 0;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (array[i][j] != 0) {
        mat->row_indices[k] = i;
        mat->col_indices[k] = j;
        mat->values[k] = array[i][j];
        k++;
      }
    }
  }
  mat->nnz = nnz;
  return mat;
}

DoubleSparseMatrix *dms_get_row(const DoubleSparseMatrix *mat, size_t i) {
  if (i >= mat->rows) {
    perror("Error: invalid row index.\n");
    return NULL;
  }
  // nzz_row = number of non-zero elements in row i
  size_t nnz_row = 0;
  for (size_t j = 0; j < mat->nnz; j++) {
    if (mat->row_indices[j] == i) {
      nnz_row++;
    }
  }

  DoubleSparseMatrix *row = dms_create(1, mat->cols, nnz_row + 1);
  size_t k = 0;
  for (size_t j = 0; j < mat->nnz; j++) {
    if (mat->row_indices[j] == i) {
      row->row_indices[k] = 0;
      row->col_indices[k] = mat->col_indices[j];
      row->values[k] = mat->values[j];
      k++;
    }
  }
  row->nnz = k;
  return row;
}

DoubleSparseMatrix *dms_get_last_row(const DoubleSparseMatrix *mat) {
  return dms_get_row(mat, mat->rows - 1);
}

DoubleSparseMatrix *dms_get_col(const DoubleSparseMatrix *mat, size_t j) {
  DoubleSparseMatrix *col = dms_create(mat->rows, 1, mat->rows);
  size_t k = 0;
  for (size_t i = 0; i < mat->nnz; i++) {
    if (mat->col_indices[i] == j) {
      col->row_indices[k] = mat->row_indices[i];
      col->col_indices[k] = 0;
      col->values[k] = mat->values[i];
      k++;
    }
  }
  col->nnz = k;
  return col;
}

DoubleSparseMatrix *dms_get_last_col(const DoubleSparseMatrix *mat) {
  return dms_get_col(mat, mat->cols - 1);
}

DoubleSparseMatrix *dms_multiply(const DoubleSparseMatrix *mat1,
                                 const DoubleSparseMatrix *mat2) {
  if (mat1->cols != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  // use cs_multiply from csparse
  cs *A = dms_to_cs(mat1);
  cs *B = dms_to_cs(mat2);
  cs *C = cs_multiply(A, B);
  DoubleSparseMatrix *result = cs_to_dms(C);

  cs_spfree(A);
  cs_spfree(B);
  cs_spfree(C);

  return result;
}

DoubleSparseMatrix *dms_multiply_by_number(const DoubleSparseMatrix *mat,
                                           const double number) {
  DoubleSparseMatrix *result = dms_create_clone(mat);
  for (size_t i = 0; i < mat->nnz; i++) {
    result->values[i] *= number;
  }
  return result;
}

DoubleSparseMatrix *dms_transpose(const DoubleSparseMatrix *mat) {
  if (mat->col_indices == NULL || mat->row_indices == NULL ||
      mat->values == NULL) {
    perror("Error: matrix is empty.\n");
    return NULL;
  }

  if (mat->nnz == 0) {
    return dms_create(mat->cols, mat->rows, 0);
  }
  // use cs_transpose from csparse
  cs *A = dms_to_cs(mat);
  cs *AT = cs_transpose(A, 1);
  DoubleSparseMatrix *result = cs_to_dms(AT);
  return result;
}

double dms_get(const DoubleSparseMatrix *mat, size_t i, size_t j) {
  if (i >= mat->rows || j >= mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return 0.0;
  }
  for (size_t k = 0; k < mat->nnz; k++) {
    if (mat->row_indices[k] == i && mat->col_indices[k] == j) {
      return mat->values[k];
    }
  }
  return 0.0;
}

void dms_set(DoubleSparseMatrix *matrix, size_t i, size_t j, double value) {
  // Find the position of the element (i, j) in the matrix
  size_t position = __dms_binary_search(matrix, i, j);

  if (position < matrix->nnz && matrix->row_indices[position] == i &&
      matrix->col_indices[position] == j) {
    // Element already exists at position (i, j), update the value
    matrix->values[position] = value;
  } else {
    __dms_insert_element(matrix, i, j, value, position);
  }
}

void dms_realloc(DoubleSparseMatrix *mat, size_t new_capacity) {
  if (new_capacity <= mat->capacity) {
    printf("Can not resize matrix to smaller capacity!\n");
    exit(EXIT_FAILURE);
  }

  if (new_capacity == 0) {
    new_capacity = mat->capacity * 2;
  }

  // resize matrix:
  size_t *row_indices =
      (size_t *)realloc(mat->row_indices, (new_capacity) * sizeof(size_t));
  size_t *col_indices =
      (size_t *)realloc(mat->col_indices, (new_capacity) * sizeof(size_t));
  double *values =
      (double *)realloc(mat->values, (new_capacity) * sizeof(double));
  if (row_indices == NULL || col_indices == NULL || values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  mat->capacity = new_capacity;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
}

void dms_print(const DoubleSparseMatrix *mat) {
  if (mat->cols > 100 || mat->rows > 100) {
    printf("Matrix is too large to print\n");
    return;
  }
  if (mat->nnz == 0) {
    printf("Empty matrix\n");
    return;
  }
  printf("Matrix: %zu x %zu\n", mat->rows, mat->cols);
  if (mat->cols <= 100 && mat->rows <= 100) {
    for (size_t i = 0; i < mat->rows; i++) {
      for (size_t j = 0; j < mat->cols; j++) {
        printf("%.2lf ", dms_get(mat, i, j));
      }
      printf("\n");
    }
  } else {
    printf("Matrix is too large to print\n");
    printf("values: [");
    for (size_t i = 0; i < mat->nnz; i++) {
      printf("%.2lf, ", mat->values[i]);
    }
    printf("]\n");

    printf("row_indices: [");
    if (mat->row_indices != NULL) {
      for (size_t i = 0; i < mat->nnz; i++) {
        printf("%zu, ", mat->row_indices[i]);
      }
    }
    printf("]\n");

    printf("col_indices: [");
    if (mat->col_indices != NULL) {
      for (size_t i = 0; i < mat->nnz; i++) {
        printf("%zu, ", mat->col_indices[i]);
      }
    }

    printf("]\n");
  }
}

void dms_destroy(DoubleSparseMatrix *mat) {
  free(mat->row_indices);
  free(mat->col_indices);
  free(mat->values);
  free(mat);
}

double *dms_to_array(const DoubleSparseMatrix *mat) {
  double *array = (double *)malloc(mat->rows * mat->cols * sizeof(double));
  if (array == NULL) {
    perror("Error: unable to allocate memory for array.\n");
    return NULL;
  }

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      array[i * mat->cols + j] = dms_get(mat, i, j);
    }
  }

  return array;
}

double dms_density(const DoubleSparseMatrix *mat) {
  return (double)mat->nnz / (double)(mat->rows * mat->cols);
}
