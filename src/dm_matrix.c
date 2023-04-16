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

#include "dm_matrix.h"

#include <assert.h>

#include "dbg.h"

// #define NDEBUG
enum { INIT_CAPACITY = 2U };

/*******************************/
/*        Double Matrix        */
/*******************************/

/**
 * @brief create an empty Double Matrix Object
 *
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_matrix() {
  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->cols = 0U;
  matrix->rows = 0U;
  matrix->values = (double *)malloc(INIT_CAPACITY * sizeof(double));
  return matrix;
}

/**
 * @brief Create a zero Double Matrix object
 *
 * @param rows
 * @param cols
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create(size_t rows, size_t cols) {

  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->values =
      (double *)malloc((matrix->rows * matrix->cols) * sizeof(double));
  return matrix;
}

/**
 * @brief Create a Random Double Matrix object
 *
 * @param num_rows
 * @param num_cols
 * @return double**
 */
DoubleMatrix *dm_create_rand(size_t rows, size_t cols) {
  DoubleMatrix *mat = dm_create(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      dm_set(mat, i, j, randomDouble());
    }
  }

  return mat;
}

/**
 * @brief Create a Identity object
 *
 * @param rows
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_identity(size_t rows) {
  DoubleMatrix *mat = dm_create(rows, rows);
  for (size_t i = 0; i < rows; i++) {
    dm_set(mat, i, i, 1);
  }
  return mat;
}

/**
 * @brief Set the Array To Matrix object
 *
 * @param rows
 * @param cols
 * @param array
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols,
                                   double array[rows][cols]) {
  DoubleMatrix *mat = dm_create(rows, cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, array[i][j]);
    }
  }

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

bool dm_is_vector(DoubleMatrix *mat) {
  if (mat->rows == 1 || mat->cols == 1) {
    return true; // Matrix is a vector
  }
  return false; // Matrix is not a vector
}

/**
 * @brief Resize the matrix
 *
 * @param rows
 * @param cols
 * @param mat
 */

void dm_resize(DoubleMatrix *mat, size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    perror("destroy matrix instead of setting zero sizes!");
  } else {
    // in case of a dense matrix:
    if ((mat->cols != cols) || (mat->rows != rows)) {
      double *new_data = (double *)calloc(rows * cols, sizeof(double));
      size_t min_rows = mat->rows < rows ? mat->rows : rows;
      size_t min_cols = mat->cols < cols ? mat->cols : cols;
      for (size_t i = 0; i < min_rows; i++) {
        for (size_t j = 0; j < min_cols; j++) {
          new_data[i * cols + j] = mat->values[i * mat->cols + j];
        }
      }
      free(mat->values);
      mat->values = new_data;
      mat->rows = rows;
      mat->cols = cols;
    }
  }
}

/**
 * @brief push (add) a column vector to  matrix
 *
 * @param mat
 * @param col_vec
 */
void dm_push_column(DoubleMatrix *mat, DoubleVector *col_vec) {
  if (mat->rows != col_vec->rows) {
    perror("Error: Length of vector does not fit to number or matrix rows");
  } else {
    // resize the matrix:
    dm_resize(mat, mat->rows, mat->cols + 1);
    for (size_t i = 0; i < mat->rows; i++) {
      dm_set(mat, i, mat->cols - 1, dv_get(col_vec, i));
    }
  }
}

/**
 * @brief push (add) a row vector to matrix
 *
 * @param mat
 * @param row_vec
 */
void dm_push_row(DoubleMatrix *mat, DoubleVector *row_vec) {
  if (row_vec->rows != mat->cols) {
    perror("Error: length of vector does not fit to number or matrix columns");

  } else {
    // resize the matrix:
    dm_resize(mat, mat->rows + 1, mat->cols);
    for (size_t i = 0; i < mat->cols; i++) {
      dm_set(mat, mat->rows - 1, i, dv_get(row_vec, i));
    }
  }
}

/**
 * @brief get value of index i, j
 *
 * @param mat
 * @param i,j
 * @return double
 */
double dm_get(const DoubleMatrix *mat, size_t i, size_t j) {

  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return 0;
  }
  return mat->values[i * mat->cols + j];
}

/**
 * @brief set value of index i, j
 *
 * @param mat
 * @param i,j
 * @param value
 */
void dm_set(DoubleMatrix *mat, size_t i, size_t j, const double value) {
  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return;
  }
  mat->values[i * mat->cols + j] = value;
}

/**
 * @brief free memory of DoubleMatrix
 *
 * @param mat
 */
void dm_destroy(DoubleMatrix *mat) {
  free(mat->values);
  free(mat);
}

/**
 * @brief Get the Row Vector object of Row row
 *
 * @param mat
 * @param row
 * @return DoubleVector
 */
DoubleVector *dv_get_row_matrix(DoubleMatrix *mat, size_t row) {
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
DoubleVector *dv_get_column_matrix(DoubleMatrix *mat, size_t column) {
  if (column < 0 || column > (mat->cols) - 1) {
    perror("This column does not exist");
  }
  DoubleVector *vec = dv_create(mat->rows);
  for (size_t i = 0; i < mat->rows; i++) {
    dv_set(vec, i, dm_get(mat, i, column));
  }

  return vec;
}

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
  vec->values = (double *)malloc(1 * sizeof(double));
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

bool dv_is_row_vector(DoubleMatrix *vec) {
  bool is_vector = false;
  if ((vec->cols == 1) && (vec->rows > 1)) {
    is_vector = true;
  }
  return is_vector;
}

/**
 * @brief Create a Double Vector Of Length object
 *
 * @param length
 * @param value
 * @return DoubleVector*
 */
DoubleVector *dv_create(size_t length) {
  DoubleVector *vec = (DoubleVector *)malloc(sizeof(DoubleVector));
  vec->rows = length;
  vec->cols = 1;
  vec->values = (double *)malloc(length * sizeof(double));
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
  double value = dm_get(vec, idx, 0);

  if (vec->rows == 1) { // only if colum-vector
    value = dm_get(vec, 0, idx);
  }

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
  if (idx < 0 || idx > (vec->rows - 1)) {
    perror("This index does not exist");
  }
  if (vec->rows == 1) {
    dm_set(vec, 0, idx, value);
  }

  dm_set(vec, idx, 0, value);
}

void dv_resize(DoubleVector *vec, size_t rows) {
  if (rows < 1) {
    perror("destroy matrix instead of setting zero sizes!");
  } else {
    // in case of a dense matrix:
    dm_resize(vec, rows, vec->cols);
  }
}

/**
 * @brief pop last column of Matrix mat
 *
 * @param mat
 * @return DoubleVector*
 */
DoubleVector *dm_pop_column_matrix(DoubleMatrix *mat) {
  DoubleVector *column_vec = dv_get_column_matrix(mat, mat->cols - 1);
  dm_resize(mat, mat->cols - 1, mat->rows);
  return column_vec;
}

/**
 * @brief pop last row of Matrix mat
 *
 * @param mat
 * @return DoubleVector*
 */
DoubleVector *dm_pop_row_matrix(DoubleMatrix *mat) {
  DoubleVector *row_vec = dv_get_row_matrix(mat, mat->rows - 1);
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
 * @brief swap two elements of an vector
 *
 * @param vec*
 * @param i
 * @param j
 */
void dv_swap_elements(DoubleVector *vec, size_t idx_i, size_t idx_j) {
  double tmp = dv_get(vec, idx_i);
  dv_set(vec, idx_i, dv_get(vec, idx_j));
  dv_set(vec, idx_j, tmp);
}




