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

#include "matrix.h"

#include "pprint2D.h"
#include "utils.h"

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
  // matrix->values[0] = calloc(sizeof(double), 0u);

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
  if (rows <= 0 || cols <= 0) {
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
      mat->values[i][j] = randomNumber();
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
DoubleMatrix* setArrayToMatrix(size_t rows, size_t cols,
                               double array[rows][cols]) {
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
  size_t old_capacity = mat->row_capacity;
  mat->row_capacity += mat->row_capacity;
  mat->values = realloc(mat->values, (mat->row_capacity) * sizeof(double*));
  for (size_t i = old_capacity; i < mat->row_capacity; i++) {
    mat->values[i] = calloc(sizeof(double), mat->column_capacity);
  }
}

/**
 * @brief extend memory in HEAP to fit changed matrix columns
 *
 * @param mat
 */
void expandMatrixColumn(DoubleMatrix* mat) {
  mat->column_capacity += mat->column_capacity;
  // mat->values = realloc(mat->values, (mat->row_capacity) * sizeof(double**));
  for (size_t i = 0; i < mat->row_capacity; i++) {
    mat->values[i] =
        realloc(mat->values[i], mat->column_capacity * sizeof(double));
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
      (mat->values[i][last_column]) = (col_vec->double_array[i]);
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
      (mat->values[last_row][i]) = (row_vec->double_array[i]);
    }

    mat->rows++;
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
 * @brief printf DoubleMatrix pretty
 *
 * @param num_cols
 * @param num_rows
 * @param matrix
 */
void printDoubleMatrix(DoubleMatrix* matrix) {
  if (matrix->rows < MAX_ROW) {
    for (size_t i = 0; i < matrix->rows; i++) {
      printDoubleArray_2(matrix->values[i], matrix->columns);
    }
  } else {
    for (size_t i = 0; i < 4; i++) {
      printDoubleArray_2(matrix->values[i], matrix->columns);
    }
    printf("...\n");
    for (size_t i = matrix->rows - 4; i < matrix->rows; i++) {
      printDoubleArray_2(matrix->values[i], matrix->columns);
    }
  }

  printf("Matrix %zix%zi, Capacity %zix%zi\n", matrix->rows, matrix->columns,
         matrix->row_capacity, matrix->column_capacity);
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
  DoubleVector* vec = createDoubleVectorOfLength(mat->columns, 0.0);
  for (size_t i = 0; i < vec->length; i++) {
    vec->double_array[i] = mat->values[row][i];
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
  DoubleVector* vec = createDoubleVectorOfLength(mat->rows, 0.0);
  for (size_t i = 0; i < mat->rows; i++) {
    vec->double_array[i] = mat->values[i][column];
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

/**
 * @brief multiply scalar with each cell of matrix
 *
 * @param mat
 * @param scalar
 */
void scalarMultiply(DoubleMatrix* mat, double scalar) {
  size_t i, j;
  for (i = 0; i < mat->columns; i++) {
    for (j = 0; j < mat->rows; j++) (mat->values[i])[j] *= scalar;
  }
}

/**
 * @brief transpose matrix mat
 *
 * @param mat
 * @return mat
 */
void transpose(DoubleMatrix* mat) {
  size_t i, j;

  for (i = 0; i < mat->columns; i++) {
    for (j = 0; j < mat->rows; j++) {
      mat->values[i][j] = mat->values[j][i];
    }
  }
}

/**
 * @brief return copy of matrix
 *
 * @param m
 * @return DoubleMatrix*
 */
DoubleMatrix* cloneMatrix(DoubleMatrix* m) {
  DoubleMatrix* copy;
  size_t i, j;
  copy = createDoubleMatrix(m->rows, m->columns);
  for (i = 0; i < m->columns; i++)
    for (j = 0; j < m->rows; j++) copy->values[i][j] = m->values[i][j];
  return copy;
}

/**
 * @brief check if two matrices are equal
 *
 * @param m1
 * @param m2
 * @return true
 * @return false
 */
bool areEqualDoubleMatrices(DoubleMatrix* m1, DoubleMatrix* m2) {
  unsigned int i, j;
  if (m1 == NULL || m2 == NULL) return false;
  if (m1->columns != m2->columns || m1->rows != m2->rows) return false;
  for (i = 0; i < m1->columns; i++) {
    for (j = 0; j < m1->rows; j++) {
      if (m1->values[i][j] != m2->values[i][j]) return false;
    }
  }
  return true;
}

/**
 * @brief Matrix Multiplication of two matrices m1 x m2
 *
 * @param m1
 * @param m2
 * @return DoubleMatrix*
 */
DoubleMatrix* multiplyDoubleMatrices(DoubleMatrix* m1, DoubleMatrix* m2) {
  size_t i, j, k;
  if (m1 == NULL || m2 == NULL) {
    perror("Error: Matrices shouldn't be empty.");
    return NULL;
  }

  if (m1->columns != m2->rows) {
    perror(
        "Error: number of columns of m1 has to be euqal to number fo rows of "
        "m2!");
    return NULL;
  }

  DoubleMatrix* product = createDoubleMatrix(m1->rows, m2->columns);

  // Multiplying first and second matrices and storing it in product
  for (i = 0; i < m1->rows; ++i) {
    for (j = 0; j < m2->columns; ++j) {
      for (k = 0; k < m1->columns; ++k) {
        product->values[i][j] += m1->values[i][k] * m2->values[k][j];
      }
    }
  }

  return product;
}

/* v1 x v2  -- simply a helper function -- computes dot product between two
 * vectors*/
double vector_multiply(double* col, double* row, size_t length) {
  double sum;
  size_t i;
  sum = 0;
  for (i = 0; i < length; i++) {
    sum += col[i] * row[i];
  }
  return sum;
}

/**
 * @brief Vector Matrix
 *
 * @param vec
 * @param mat
 * @return DoubleVector*
 */
DoubleVector* multiplyVectorMatrix(DoubleVector* vec, DoubleMatrix* mat) {
  DoubleVector* vec_result = createDoubleVectorOfLength(vec->length, 0.0);
  for (size_t i = 0; i < vec->length; i++) {
    vec_result->double_array[i] =
        vector_multiply(mat->values[i], vec->double_array, vec->length);
  }

  return vec_result;
}
