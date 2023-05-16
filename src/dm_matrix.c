/**
 * @file dm_matrix.c
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

// #define NDEBUG
enum { INIT_CAPACITY = 1000U };

/*******************************/
/*        Double Matrix        */
/*******************************/

/**
 * @brief create a Double Matrix Object
 *
 * @param rows
 * @param cols
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create(size_t rows, size_t cols) {
  return dm_create_format(rows, cols, SPARSE);
}

/**
 * @brief create a Double Matrix Object with given format:
 *
 * @param rows
 * @param cols
 * @param format (SPARSE, DENSE, VECTOR)
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_format(size_t rows, size_t cols, matrix_format format) {
  DoubleMatrix *mat = NULL;

  switch (format) {
  case SPARSE:
    mat = dm_create_sparse(rows, cols);
    break;
  case DENSE:
    mat = dm_create_dense(rows, cols);
    break;
  case VECTOR:
    mat = dv_create(rows);
    break;
  default:
    perror("Error: invalid matrix format.\n");
    return NULL;
  }

  return mat;
}

/**
 * @brief create a sparse DoubleMatrix Object size to fit nnz elements
 *
 * @param rows
 * @param cols
 * @param size
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_nnz(size_t rows, size_t cols, size_t nnz) {
  DoubleMatrix *mat = dm_create_sparse(rows, cols);
  dm_realloc_sparse(mat, nnz);
  return mat;
}

/**
 * @brief return copy of matrix
 *
 * @param m
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_clone(DoubleMatrix *mat) {
  DoubleMatrix *copy = dm_create(mat->rows, mat->cols);
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(copy, i, j, dm_get(mat, i, j));
    }
  }
  return copy;
}

/**
 * @brief get value of index i, j
 *
 * @param mat
 * @param i,j
 * @return double
 */
double dm_get(const DoubleMatrix *mat, size_t i, size_t j) {
  // perror if boundaries are exceeded
  if (i >= mat->rows || j >= mat->cols) {
    perror("Error: index out of bounds.\n");
    // dbg(i);
    // dbg(j);
  }
  switch (mat->format) {
  case DENSE:
    return dm_get_dense(mat, i, j);
    break;
  case SPARSE:
    return dm_get_sparse(mat, i, j);
    break;
  case VECTOR:
    return dv_get(mat, i);
    break;
  }
}

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
  case VECTOR:
    dv_set(mat, i, value);
    break;
  }
}

// free sparse matrix
void dm_destroy(DoubleMatrix *mat) {
  free(mat->col_indices);
  free(mat->values);
  free(mat->row_indices);
  free(mat);
  mat = NULL;
}

/*******************************/
/*    private functions        */
/*******************************/

static DoubleMatrix *dm_create_sparse(size_t rows, size_t cols) {
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

static DoubleMatrix *dm_create_dense(size_t rows, size_t cols) {

  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = 0;
  matrix->format = DENSE;
  matrix->capacity = 0;
  matrix->row_indices = NULL;
  matrix->col_indices = NULL;
  matrix->values = (double *)calloc(rows * cols, sizeof(double));
  return matrix;
}

static void dm_set_sparse(DoubleMatrix *mat, size_t i, size_t j, double value) {
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

// get value from dense matrix:
static void dm_set_dense(DoubleMatrix *mat, size_t i, size_t j,
                         const double value) {
  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return;
  }
  mat->values[i * mat->cols + j] = value;
  mat->nnz++;
}

// get value from dense matrix:
static double dm_get_dense(const DoubleMatrix *mat, size_t i, size_t j) {
  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return 0.0;
  }
  return mat->values[i * mat->cols + j];
}

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
