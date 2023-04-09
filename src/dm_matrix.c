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
DoubleMatrix *new_dm_matrix() {
  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->columns = 0U;
  matrix->rows = 0U;
  matrix->columnCapacity = INIT_CAPACITY;
  matrix->rowCapacity = INIT_CAPACITY;

  matrix->values = (double **)malloc(sizeof(double *));
  matrix->values[0] = calloc(sizeof(double), matrix->columnCapacity);

  return matrix;
}

/**
 * @brief Create a zero Double Matrix object
 *
 * @param rows
 * @param cols
 * @return DoubleMatrix*
 */
DoubleMatrix *create_dm_matrix(size_t rows, size_t cols) {

  DoubleMatrix *matrix = (DoubleMatrix *)malloc(sizeof(DoubleMatrix));
  matrix->rows = rows;
  matrix->columns = cols;

  if (rows > INIT_CAPACITY) {
    matrix->rowCapacity = rows + INIT_CAPACITY;
  } else {
    matrix->rowCapacity = INIT_CAPACITY;
  }

  if (cols > INIT_CAPACITY) {
    matrix->columnCapacity = cols + INIT_CAPACITY;
  } else {
    matrix->columnCapacity = INIT_CAPACITY;
  }

  matrix->values = (double **)malloc((matrix->rowCapacity) * sizeof(double *));
  for (size_t i = 0; i < matrix->rowCapacity; i++) {
    matrix->values[i] = calloc(sizeof(double), matrix->columnCapacity);
  }

  return matrix;
}

/**
 * @brief Create a Random Double Matrix object
 *
 * @param num_rows
 * @param num_cols
 * @return double**
 */
DoubleMatrix *create_rand_dm_matrix(size_t rows, size_t cols) {
  DoubleMatrix *mat = create_dm_matrix(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat->values[i][j] = randomDouble();
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
DoubleMatrix *create_identity_matrix(size_t rows) {
  DoubleMatrix *mat = create_dm_matrix(rows, rows);
  for (size_t i = 0; i < rows; i++) {
    (mat->values[i])[i] = 1;
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
DoubleMatrix *set_array_to_dm_matrix(size_t rows, size_t cols, double **array) {
  DoubleMatrix *mat = create_dm_matrix(rows, cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->columns; j++) {
      mat->values[i][j] = array[i][j];
    }
  }

  return mat;
}

/**
 * @brief extend memory in HEAP to fit changed matrix rows
 *
 * @param mat
 */
static void expand_dm_matrix_row(DoubleMatrix *mat) {
  // size_t old_capacity = mat->rowCapacity;
  mat->rowCapacity += 1;
  mat->values = realloc(mat->values, (mat->rowCapacity) * sizeof(double *));
  for (size_t i = 0; i < mat->rowCapacity; i++) {
    mat->values[i] =
        realloc(mat->values[i], mat->columnCapacity * sizeof(double));
  }
}

/**
 * @brief extend memory in HEAP to fit changed matrix columns
 *
 * @param mat
 */
static void expand_dm_matrix_column(DoubleMatrix *mat) {
  mat->columnCapacity += mat->columnCapacity;
  mat->values = realloc(mat->values, (mat->rowCapacity) * sizeof(double *));
  for (size_t i = 0; i < mat->rowCapacity; i++) {
    mat->values[i] = realloc(mat->values[i], mat->rowCapacity * sizeof(double));
  }
}

/**
 * @brief reduce memory in HEAP if possible
 *
 * @param mat
 */
static void shrink_dm_matrix_column(DoubleMatrix *mat) {
  if ((mat->columns<(mat->columnCapacity - INIT_CAPACITY) &
                    (mat->columnCapacity - INIT_CAPACITY)> 1)) {
    mat->columnCapacity -= INIT_CAPACITY;

    for (size_t i = 0; i < mat->rowCapacity; i++) {
      mat->values[i] =
          realloc(mat->values[i], mat->columnCapacity * sizeof(double));
    }
  }
}

/**
 * @brief reduce memory in HEAP if possible
 *
 * @param mat
 */
static void shrink_dm_matrix_row(DoubleMatrix *mat) {
  if ((mat->rows<(mat->rowCapacity - INIT_CAPACITY) &
                 (mat->rowCapacity - INIT_CAPACITY)> 1)) {
    mat->rowCapacity -= INIT_CAPACITY;

    mat->values = realloc(mat->values, (mat->rowCapacity) * sizeof(double *));
  }
}

/**
 * @brief push (add) a column vector to  matrix
 *
 * @param mat
 * @param col_vec
 */
void push_column(DoubleMatrix *mat, const DoubleVector *col_vec) {
  if (col_vec->length != mat->rows) {
    perror("Error: length of vector does not fit to number or matrix rows");

  } else {
    if (mat->columns == mat->columnCapacity) {
      expand_dm_matrix_column(mat);
    }

    size_t last_column = mat->columns;
    for (size_t i = 0; i < mat->rows; i++) {
      (mat->values[i][last_column]) = (col_vec->mat1D->values[i][0]);
    }

    mat->columns++;
  }
}

/**
 * @brief push (add) a row vector to matrix
 *
 * @param mat
 * @param row_vec
 */
void push_row(DoubleMatrix *mat, const DoubleVector *row_vec) {
  if (row_vec->length != mat->columns) {
    perror("Error: length of vector does not fit to number or matrix columns");

  } else {
    if (mat->rows == mat->rowCapacity) {
      expand_dm_matrix_row(mat);
    }

    size_t last_row = mat->rows;
    for (size_t i = 0; i < mat->columns; i++) {
      (mat->values[last_row][i]) = (row_vec->mat1D->values[i][0]);
    }

    mat->rows++;
  }
}

/**
 * @brief free memory of DoubleMatrix
 *
 * @param mat
 */
void free_dm_matrix(DoubleMatrix *mat) {
  for (size_t i = 0; i < mat->rowCapacity; i++) {
    free(mat->values[i]);
  }
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
DoubleVector *get_row_vector(const DoubleMatrix *mat, size_t row) {
  if (row < 0 || row > (mat->rows - 1)) {
    perror("This row does not exist");
  }
  // INFO: chg mat->column in mat->rows
  DoubleVector *vec = new_dm_vector_length(mat->columns, 0.0);
  for (size_t i = 0; i < vec->mat1D->columns; i++) {
    vec->mat1D->values[i][0] = mat->values[row][i];
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
DoubleVector *get_column_vector(const DoubleMatrix *mat, size_t column) {
  if (column < 0 || column > (mat->columns - 1)) {
    perror("This column does not exist");
  }
  DoubleVector *vec = new_dm_vector_length(
      mat->rows, 0.0); // INFO: chg mat->rows in mat->columns
  for (size_t i = 0; i < mat->rows; i++) {
    vec->mat1D->values[i][0] = mat->values[i][column];
  }

  return vec;
}

/**
 * @brief retrun adress of row vector
 *
 * @param mat
 * @param row
 * @return DoubleVector*
 */
double *get_row_array(const DoubleMatrix *mat, size_t row) {
  if (row < 0 || row > (mat->rows - 1)) {
    perror("This row does not exist");
  }
  return (mat->values[row]);
}

/*******************************/
/*  Double Vector (Dynamic)    */
/*******************************/

/**
 * @brief Create a DoubleVector object (HEAP INIT_CAPACITY)
 * @return DoubleVector*
 */

DoubleVector *new_dm_vector() {
  DoubleVector *vec = (DoubleVector *)malloc(sizeof(DoubleVector));
  if (!vec) {
    return NULL;
  }
  vec->isColumnVector = false;
  vec->length = 0;
  vec->mat1D = new_dm_matrix();
  return vec;
}

DoubleVector *dv_create_from_array(const double *array, const size_t length) {
  DoubleVector *vec = new_dm_vector_length(length, 0.0);
  for (size_t i = 0; i < vec->length; i++) {
    vec->mat1D->values[i][0] = array[i];
  }

  return vec;
}

/**
 * @brief Clone a DoubleVector object
 * @return DoubleVector*
 */
DoubleVector *clone_dm_vector(const DoubleVector *vector) {
  size_t org_length = vector->length;
  DoubleVector *clone = new_dm_vector_length(org_length, 0.);
  for (size_t i = 0; i < org_length; i++) {
    clone->mat1D->values[i][0] = vector->mat1D->values[i][0];
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
DoubleVector *new_dm_vector_length(size_t length, double value) {
  DoubleVector *vec = (DoubleVector *)malloc(sizeof(DoubleVector));
  vec->isColumnVector = false;
  vec->length = length;

  vec->mat1D = create_dm_matrix(length, 0);
  if (vec->mat1D->values == NULL) {
    dbg(vec->mat1D);
  }
  for (size_t i = 0; i < length; i++) {
    vec->mat1D->values[i][0] = value;
    // dbg(vec->mat1D->values[i][0]);
  }
  return vec;
}

/**
 * @brief Create a Random Double Vector object
 *
 * @param length
 * @return DoubleVector
 */
DoubleVector *new_rand_dm_vector_length(size_t length) {
  DoubleVector *vec = (DoubleVector *)malloc(sizeof(DoubleVector));
  vec->isColumnVector = false;
  vec->length = length;
  vec->mat1D = create_dm_matrix(length, 0);

  for (size_t i = 0; i < length; i++) {
    vec->mat1D->values[i][0] = randomDouble();
  }

  return vec;
}

void set_dm_vector_to_array(DoubleVector *vec, const double *array,
                            size_t len_array) {
  assert(len_array > 0);
  if (vec->mat1D->values != NULL) {
    if (len_array < vec->length) {
      for (size_t i = 0; i < len_array; i++) {
        vec->mat1D->values[i][0] = array[i];
      }
      for (size_t i = len_array; i < vec->length; i++) {
        pop_value(vec);
      }
    } else if (len_array >= vec->length) {
      for (size_t i = 0; i < vec->length; i++) {
        vec->mat1D->values[i][0] = array[i];
      }
      size_t len = vec->length;
      for (size_t i = len; i < len_array; i++) {
        push_value(vec, array[i]);
      }
    }
  } else {
    dbg(vec->mat1D->values);
  }
}

/**
 * @brief pop last column of Matrix mat
 *
 * @param mat
 * @return DoubleVector*
 */
DoubleVector *pop_column(DoubleMatrix *mat) {
  DoubleVector *column_vec = get_column_vector(mat, mat->columns - 1);

  mat->columns--;

  if (mat->columns < (mat->columnCapacity - INIT_CAPACITY)) {
    shrink_dm_matrix_column(mat);
  }

  return column_vec;
}

/**
 * @brief pop last row of Matrix mat
 *
 * @param mat
 * @return DoubleVector*
 */
DoubleVector *pop_row(DoubleMatrix *mat) {
  DoubleVector *row_vec = get_row_vector(mat, mat->rows - 1);

  mat->rows--;

  if (mat->rows < (mat->rowCapacity - INIT_CAPACITY)) {
    shrink_dm_matrix_row(mat);
  }

  return row_vec;
}

/**
 * @brief get double array from values
 *
 * @param vec
 * @return double*
 */
double *get_array_from_vector(const DoubleVector *vec) {
  double *array = (double *)malloc(vec->mat1D->rowCapacity * sizeof(double));
  for (size_t i = 0; i < vec->mat1D->rows; i++) {
    array[i] = vec->mat1D->values[i][0];
  }

  return array;
}

/**
 * @brief expand allocate memory in HEAP with another INIT_CAPACITY
 *
 * @param vec
 */
static void expand_dm_vector(DoubleVector *vec) {
  // Remove: printf("capacity: %zu", vec->mat1D->row_capacity);
  expand_dm_matrix_row(vec->mat1D);
}

/**
 * @brief shrink allocate memory in HEAP
 *
 * @param vec
 */
static void shrink_dm_vector(DoubleVector *vec) {
  shrink_dm_matrix_row(vec->mat1D);
}

/**
 * @brief push (add) new value to vector vec
 *
 * @param vec
 * @param value
 */
void push_value(DoubleVector *vec, double value) {
  vec->length = vec->length + 1;
  if (vec->length >= vec->mat1D->rowCapacity) {
    expand_dm_vector(vec);
  }
  vec->mat1D->values[vec->mat1D->rows][0] = value;
  vec->mat1D->rows += 1;
}

/**
 * @brief pop (get) last element if DoubleVector vec
 *
 * @param vec
 * @return double
 */
double pop_value(DoubleVector *vec) {
  double value = vec->mat1D->values[vec->mat1D->rows - 1][0];

  vec->mat1D->values[vec->length][0] = 0.0;
  vec->length--;
  if (vec->length <= vec->mat1D->rowCapacity) {
    shrink_dm_vector(vec);
  }

  return value;
}

/**
 * @brief free memory of DoubleVector
 *
 * @param vec
 * @return DoubleVector*
 */
void free_dm_vector(DoubleVector *vec) {
  free_dm_matrix(vec->mat1D);
  vec->mat1D = NULL;
  free(vec);
  vec = NULL;
}

/**
 * @brief swap two elements of an vector
 *
 * @param vec*
 * @param i
 * @param j
 */
void swap_elements_vector(DoubleVector *vec, size_t idx_i, size_t idx_j) {
  double tmp = vec->mat1D->values[idx_i][0];
  vec->mat1D->values[idx_i][0] = vec->mat1D->values[idx_j][0];
  vec->mat1D->values[idx_j][0] = tmp;
}

/**
 * @brief reverse the order of elements of vec
 *
 * @param vec*
 */
void reverse_vector(DoubleVector *vec) {
  double temp = 0;

  for (size_t i = 0; i < vec->mat1D->rows / 2; i++) {
    temp = vec->mat1D->values[i][0];
    vec->mat1D->values[i][0] = vec->mat1D->values[vec->mat1D->rows - i - 1][0];
    vec->mat1D->values[vec->mat1D->rows - i - 1][0] = temp;
  }
}
