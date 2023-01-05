/**
 * @file matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 26-12-2ß22
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm_matrix.h"

#include <assert.h>

#include "dbg.h"

#define INIT_CAPACITY 2u

/*******************************/
/*        Double Matrix        */
/*******************************/

/**
 * @brief create an empty Double Matrix Object
 *
 * @return DoubleMatrix*
 */
DoubleMatrix* newDoubleMatrix() {
  DoubleMatrix* matrix = (DoubleMatrix*)malloc(sizeof(DoubleMatrix));
  matrix->columns = 0u;
  matrix->rows = 0u;
  matrix->column_capacity = INIT_CAPACITY;
  matrix->row_capacity = INIT_CAPACITY;

  matrix->values = (double**)malloc(sizeof(double*));
  matrix->values[0] = calloc(sizeof(double), matrix->column_capacity);

  return matrix;
}

/**
 * @brief Create a zero Double Matrix object
 *
 * @param rows
 * @param cols
 * @return DoubleMatrix*
 */
DoubleMatrix* createDoubleMatrix(size_t rows, size_t cols) {
  if (rows < 0 || cols < 0) {
    perror("Dimension has to be positive");
    return NULL;
  }
  DoubleMatrix* matrix = (DoubleMatrix*)malloc(sizeof(DoubleMatrix));
  matrix->rows = rows;
  matrix->columns = cols;

  if (rows > INIT_CAPACITY) {
    matrix->row_capacity = rows + INIT_CAPACITY;
  } else {
    matrix->row_capacity = INIT_CAPACITY;
  }

  if (cols > INIT_CAPACITY) {
    matrix->column_capacity = cols + INIT_CAPACITY;
  } else {
    matrix->column_capacity = INIT_CAPACITY;
  }

  matrix->values = (double**)malloc((matrix->row_capacity) * sizeof(double*));
  for (size_t i = 0; i < matrix->row_capacity; i++) {
    matrix->values[i] = calloc(sizeof(double), matrix->column_capacity);
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
DoubleMatrix* createRandomDoubleMatrix(size_t rows, size_t cols) {
  DoubleMatrix* mat = createDoubleMatrix(rows, cols);

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
DoubleMatrix* createIdentityMatrix(size_t rows) {
  DoubleMatrix* mat = createDoubleMatrix(rows, rows);
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
DoubleMatrix* setArrayToMatrix(size_t rows, size_t cols, double** array) {
  DoubleMatrix* mat = createDoubleMatrix(rows, cols);

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
void expandMatrixRow(DoubleMatrix* mat) {
  // size_t old_capacity = mat->row_capacity;
  mat->row_capacity += 1;
  mat->values = realloc(mat->values, (mat->row_capacity) * sizeof(double*));
  for (size_t i = 0; i < mat->row_capacity; i++) {
    mat->values[i] =
        realloc(mat->values[i], mat->column_capacity * sizeof(double));
  }
}

/**
 * @brief extend memory in HEAP to fit changed matrix columns
 *
 * @param mat
 */
void expandMatrixColumn(DoubleMatrix* mat) {
  mat->column_capacity += mat->column_capacity;
  mat->values = realloc(mat->values, (mat->row_capacity) * sizeof(double*));
  for (size_t i = 0; i < mat->row_capacity; i++) {
    mat->values[i] =
        realloc(mat->values[i], mat->row_capacity * sizeof(double));
  }
}

/**
 * @brief reduce memory in HEAP if possible
 *
 * @param mat
 */
void shrinkMatrixColumn(DoubleMatrix* mat) {
  if ((mat->columns<(mat->column_capacity - INIT_CAPACITY) &
                    (mat->column_capacity - INIT_CAPACITY)> 1)) {
    mat->column_capacity -= INIT_CAPACITY;

    for (size_t i = 0; i < mat->row_capacity; i++) {
      mat->values[i] =
          realloc(mat->values[i], mat->column_capacity * sizeof(double));
    }
  }
}

/**
 * @brief reduce memory in HEAP if possible
 *
 * @param mat
 */
void shrinkMatrixRow(DoubleMatrix* mat) {
  if ((mat->rows<(mat->row_capacity - INIT_CAPACITY) &
                 (mat->row_capacity - INIT_CAPACITY)> 1)) {
    mat->row_capacity -= INIT_CAPACITY;

    mat->values = realloc(mat->values, (mat->row_capacity) * sizeof(double*));
  }
}

/**
 * @brief push (add) a column vector to  matrix
 *
 * @param mat
 * @param col_vec
 */
void pushColumn(DoubleMatrix* mat, DoubleVector* col_vec) {
  if (col_vec->length != mat->rows) {
    perror("Error: length of vector does not fit to number or matrix rows");

  } else {
    if (mat->columns == mat->column_capacity) {
      expandMatrixColumn(mat);
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
void pushRow(DoubleMatrix* mat, DoubleVector* row_vec) {
  if (row_vec->length != mat->columns) {
    perror("Error: length of vector does not fit to number or matrix columns");

  } else {
    if (mat->rows == mat->row_capacity) {
      expandMatrixRow(mat);
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
void freeDoubleMatrix(DoubleMatrix* mat) {
  for (size_t i = 0; i < mat->columns; i++) free(mat->values[i]);
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
DoubleVector* getRowVector(DoubleMatrix* mat, size_t row) {
  if (row < 0 || row > (mat->rows - 1)) {
    perror("This row does not exist");
  }
  // INFO: chg mat->column in mat->rows
  DoubleVector* vec = newDoubleVectorOfLength(mat->rows, 0.0);
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
DoubleVector* getColumnVector(DoubleMatrix* mat, size_t column) {
  if (column < 0 || column > (mat->columns - 1)) {
    perror("This column does not exist");
  }
  DoubleVector* vec = newDoubleVectorOfLength(
      mat->columns, 0.0);  // INFO: chg mat->rows in mat->columns
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
double* showRowVector(DoubleMatrix* mat, size_t row) {
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

DoubleVector* newDoubleVector() {
  DoubleVector* vec = (DoubleVector*)malloc(sizeof(DoubleVector));
  if (!vec) return NULL;
  vec->column_vec = false;
  vec->length = 0;
  vec->mat1D = newDoubleMatrix();
  return vec;
}

/**
 * @brief Clone a DoubleVector object
 * @return DoubleVector*
 */
DoubleVector* cloneDoubleVector(const DoubleVector* vector) {
  size_t org_length = vector->length;
  DoubleVector* clone = newDoubleVectorOfLength(org_length, 0.);
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
DoubleVector* newDoubleVectorOfLength(size_t length, double value) {
  DoubleVector* vec = (DoubleVector*)malloc(sizeof(DoubleVector));
  vec->column_vec = false;
  vec->length = length;

  vec->mat1D = createDoubleMatrix(length, 0);
  if (vec->mat1D->values != NULL) {
    for (size_t i = 0; i < length; i++) {
      vec->mat1D->values[i][0] = value;
      // dbg(vec->mat1D->values[i][0]);
    }

  } else {
    dbg(vec->mat1D->values[0][0]);
  }
  return vec;
}

/**
 * @brief Create a Random Double Vector object
 *
 * @param length
 * @return DoubleVector
 */
DoubleVector* newRandomDoubleVectorOfLength(size_t length) {
  DoubleVector* vec = (DoubleVector*)malloc(sizeof(DoubleVector));
  vec->column_vec = false;
  vec->length = length;
  vec->mat1D = createDoubleMatrix(length, 0);

  for (size_t i = 0; i < length; i++) {
    vec->mat1D->values[i][0] = randomDouble();
  }

  return vec;
}

void setDoubleVectorArray(DoubleVector* vec, double* array, size_t len_array) {
  if (len_array <= 0) assert(len_array);
  if (vec->mat1D->values != NULL) {
    if (len_array < vec->length) {
      for (size_t i = 0; i < len_array; i++) {
        vec->mat1D->values[i][0] = array[i];
      }
      for (size_t i = len_array; i < vec->length; i++) {
        popValue(vec);
      }
    } else if (len_array >= vec->length) {
      for (size_t i = 0; i < vec->length; i++) {
        vec->mat1D->values[i][0] = array[i];
      }
      size_t len = vec->length;
      for (size_t i = len; i < len_array; i++) {
        pushValue(vec, array[i]);
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
DoubleVector* popColumn(DoubleMatrix* mat) {
  DoubleVector* column_vec = getColumnVector(mat, mat->columns - 1);

  mat->columns--;

  if (mat->columns < (mat->column_capacity - INIT_CAPACITY)) {
    shrinkMatrixColumn(mat);
  }

  return column_vec;
}

/**
 * @brief pop last row of Matrix mat
 *
 * @param mat
 * @return DoubleVector*
 */
DoubleVector* popRow(DoubleMatrix* mat) {
  DoubleVector* row_vec = getRowVector(mat, mat->rows - 1);

  mat->rows--;

  if (mat->rows < (mat->row_capacity - INIT_CAPACITY)) {
    shrinkMatrixRow(mat);
  }

  return row_vec;
}

/**
 * @brief get double array from values
 *
 * @param vec
 * @return double*
 */
double* getArrayFromVector(DoubleVector* vec) {
  double* array = (double*)malloc(vec->mat1D->row_capacity * sizeof(double));
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
void expandDoubleVector(DoubleVector* vec) {
  // Remove: printf("capacity: %zu", vec->mat1D->row_capacity);
  expandMatrixRow(vec->mat1D);
}

/**
 * @brief shrink allocate memory in HEAP
 *
 * @param vec
 */
void shrinkDoubleVector(DoubleVector* vec) { shrinkMatrixRow(vec->mat1D); }

/**
 * @brief push (add) new value to vector vec
 *
 * @param vec
 * @param value
 */
void pushValue(DoubleVector* vec, double value) {
  vec->length = vec->length + 1;
  if (vec->length >= vec->mat1D->row_capacity) {
    expandDoubleVector(vec);
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
double popValue(DoubleVector* vec) {
  double value;
  value = vec->mat1D->values[vec->mat1D->rows - 1][0];

  vec->mat1D->values[vec->length][0] = 0.0;
  vec->length--;
  if (vec->length <= vec->mat1D->row_capacity) {
    shrinkDoubleVector(vec);
  }

  return value;
}

/**
 * @brief free memory of DoubleVector
 *
 * @param vec
 * @return DoubleVector*
 */
void freeDoubleVector(DoubleVector* vec) {
  freeDoubleMatrix(vec->mat1D);
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
void swapElementsOfVector(DoubleVector* vec, size_t i, size_t j) {
  double tmp = vec->mat1D->values[i][0];
  vec->mat1D->values[i][0] = vec->mat1D->values[j][0];
  vec->mat1D->values[j][0] = tmp;
}

/**
 * @brief reverse the order of elements of vec
 *
 * @param vec*
 */
void reverseVector(DoubleVector* vec) {
  double temp;

  for (size_t i = 0; i < vec->mat1D->rows / 2; i++) {
    temp = vec->mat1D->values[i][0];
    vec->mat1D->values[i][0] = vec->mat1D->values[vec->mat1D->rows - i - 1][0];
    vec->mat1D->values[vec->mat1D->rows - i - 1][0] = temp;
  }
}
