/**
 * @file dv_vector.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 17-04-2023
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "dv_vector.h"
#include "dbg.h"
#include "dm.h"
#include "dm_internals.h"
#include "dm_math.h"

enum { INIT_CAPACITY = 2U };

/*******************************/
/*  Double Vector (Dynamic)    */
/*******************************/

/**
 * @brief Create a DoubleVector object (HEAP INIT_CAPACITY)
 * @return DoubleVector*
 */

DoubleVector *dv_vector() {
  DoubleVector *vec = (DoubleVector *)malloc(sizeof(DoubleVector));
  vec->rows = 0;
  vec->cols = 1;
  vec->nnz = 0;
  vec->capacity = INIT_CAPACITY;
  vec->row_indices = NULL;
  vec->col_indices = NULL;
  vec->format = VECTOR;
  vec->values = (double *)malloc(vec->capacity * sizeof(double));
  vec->hash_table = NULL;
  return vec;
}

DoubleVector *dv_create_from_array(const double *array, const size_t length) {
  DoubleVector *vec = dv_create(length);
  for (size_t i = 0; i < length; i++) {
    dv_set(vec, i, array[i]);
  }

  return vec;
}

/**
 * @brief Clone a DoubleVector object
 * @return DoubleVector*
 */
DoubleVector *dv_clone(DoubleVector *vector) {
  DoubleVector *clone = dv_create(vector->rows);
  for (size_t i = 0; i < vector->rows; i++) {
    dv_set(clone, i, dv_get(vector, i));
  }
  return clone;
}

/**
 * @brief Create a Double Vector Of Length object
 *
 * @param length
 * @param value
 * @return DoubleVector*
 */
DoubleVector *dv_create(size_t length) {
  if (length < 1) {
    perror("Vector length must be greater than 0");
  }
  DoubleVector *vec = dv_vector();
  vec->rows = length;
  vec->capacity = length + INIT_CAPACITY;

  double *values = (double *)malloc(vec->capacity * sizeof(double));
  if (values == NULL) {
    perror("Could not allocate memory for vector");
    exit(EXIT_FAILURE);
  }

  vec->values = values;
  return vec;
}

/**
 * @brief Create a Random Double Vector object
 *
 * @param length
 * @return DoubleVector
 */
DoubleVector *dv_create_rand(size_t length) {
  DoubleVector *vec = dv_create(length);

  for (size_t i = 0; i < length; i++) {
    dv_set(vec, i, randomDouble());
  }

  return vec;
}

/**
 * @brief get value of index
 *
 * @param vec
 * @param idx
 * @return double
 */
double dv_get(const DoubleVector *vec, size_t idx) {
  if (idx < 0 || idx > (vec->rows)) {
    perror("This index does not exist");
  }
  double value = vec->values[idx];

  return value;
}

/**
 * @brief set value of index
 *
 * @param vec
 * @param idx
 * @param double
 */
void dv_set(DoubleVector *vec, size_t idx, double value) {
  if ((idx < 0) || (idx > (vec->rows))) {
    perror("This index does not exist");
  }
  // set value
  vec->values[idx] = value;
}

void dv_resize(DoubleVector *vec, size_t rows) {
  if (rows < 1) {
    perror("destroy matrix instead of setting zero sizes!");
  } else {
    // in case of a dense matrix:
    dm_resize(vec, rows, 1);
  }
}

/**
 * @brief pop last column of Matrix mat
 *
 * @param mat
 * @return DoubleVector*
 */
DoubleVector *dm_pop_column_vector(DoubleMatrix *mat) {
  DoubleVector *column_vec = dv_get_column_vector(mat, mat->cols - 1);
  dm_resize(mat, mat->cols - 1, mat->rows);
  return column_vec;
}

/**
 * @brief pop last row of Matrix mat
 *
 * @param mat
 * @return DoubleVector*
 */
DoubleVector *dm_pop_row_vector(DoubleMatrix *mat) {
  DoubleVector *row_vec = dv_get_row_vector(mat, mat->rows - 1);
  dm_resize(mat, mat->rows - 1, mat->cols);
  return row_vec;
}

/**
 * @brief get double array from values
 *
 * @param vec
 * @return double*
 */
double *dv_get_array(const DoubleVector *vec) { return vec->values; }

/**
 * @brief push (add) new value to vector vec
 *
 * @param vec
 * @param value
 */
void dv_push_value(DoubleVector *vec, double value) {
  dv_resize(vec, vec->rows + 1);
  dv_set(vec, vec->rows - 1, value);
}

/**
 * @brief pop (get) last element if DoubleVector vec
 *
 * @param vec
 * @return double
 */
double dv_pop_value(DoubleVector *vec) {
  double value = dv_get(vec, vec->rows - 1);
  dv_resize(vec, vec->rows - 1);
  return value;
}

/**
 * @brief free memory of DoubleVector
 *
 * @param vec
 * @return DoubleVector*
 */
void dv_destroy(DoubleVector *vec) {
  // in case of a dense matrix:
  dm_destroy(vec);
}

/**
 * @brief Get the Row Vector object of Row row
 *
 * @param mat
 * @param row
 * @return DoubleVector
 */
DoubleVector *dv_get_row_vector(DoubleMatrix *mat, size_t row) {
  if (row < 0 || row > (mat->rows - 1)) {
    perror("This row does not exist");
  }
  DoubleVector *vec = dv_create(mat->cols);
  for (size_t i = 0; i < vec->rows; i++) {
    dv_set(vec, i, dm_get(mat, row, i));
  }

  return vec;
}

/**
 * @brief Get the Column Vector object
 *
 * @param mat
 * @param column
 * @return DoubleVector
 */
DoubleVector *dv_get_column_vector(DoubleMatrix *mat, size_t column) {
  if (column < 0 || column > (mat->cols) - 1) {
    perror("This column does not exist");
  }
  DoubleVector *vec = dv_create(mat->rows);
  for (size_t i = 0; i < mat->rows; i++) {
    dv_set(vec, i, dm_get(mat, i, column));
  }

  return vec;
}
