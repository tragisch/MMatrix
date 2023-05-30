/**
 * @file dm_create.c
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

// #define NDEBUG
enum { INIT_CAPACITY = 1000U };

/*******************************/
/*        DEFAULT FORMAT       */
/*******************************/

matrix_format default_matrix_format = COO; // default format

/**
 * @brief Set the Default Matrix Format object
 *
 * @param format
 */

void set_default_matrix_format(matrix_format format) {
  default_matrix_format = format;
}

/*******************************/
/*        DECONSTRUCTOR        */
/*******************************/

// free sparse matrix
void dm_destroy(DoubleMatrix *mat) {

  free(mat->col_indices);
  free(mat->values);
  free(mat->row_indices);
  free(mat);
  mat = NULL;
}

/*******************************/
/*        CONSTRUCTORS         */
/*******************************/

/**
 * @brief create a Double Matrix Object with given format:
 *
 * @param rows
 * @param cols
 * @param format (COO, CSR, DENSE, VECTOR)
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_format(size_t rows, size_t cols, matrix_format format) {
  DoubleMatrix *mat = NULL;

  switch (format) {
  case COO:
    mat = dm_create_coo(rows, cols);
    break;
  case CSR:
    // mat = dm_create_CSR(rows, cols);
    break;
  case DENSE:
    mat = dm_create_dense(rows, cols);
    break;
  case VECTOR:
    if (rows != 1 && cols != 1) {
      perror("Error: vector must have one dimension of size 1.\n");
      exit(EXIT_FAILURE);
    }
    mat = dv_create(rows > cols ? rows : cols);
    break;
  default:
    perror("Error: invalid matrix format.\n");
    return NULL;
  }

  return mat;
}

/**
 * @brief create a Double Matrix Object
 *
 * @param rows
 * @param cols
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create(size_t rows, size_t cols) {
  return dm_create_format(rows, cols, default_matrix_format);
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
  DoubleMatrix *mat = dm_create_format(rows, cols, default_matrix_format);
  if (mat->format == COO) {
    dm_realloc_sparse(mat, nnz);
  }
  return mat;
}

/**
 * @brief return copy of matrix
 *
 * @param m
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_clone(const DoubleMatrix *mat) {
  DoubleMatrix *copy = dm_create_format(mat->rows, mat->cols, mat->format);
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(copy, i, j, dm_get(mat, i, j));
    }
  }
  return copy;
}

/*******************************/
/*        COO MATRIX        */
/*******************************/

static DoubleMatrix *dm_create_coo(size_t rows, size_t cols) {
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
  mat->format = COO;
  mat->values = calloc(max_int(INIT_CAPACITY, (int)mat->nnz), sizeof(double));
  return mat;
}

/*******************************/
/*        DENSE MATRIX         */
/*******************************/

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
