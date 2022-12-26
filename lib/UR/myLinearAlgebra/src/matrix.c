/**
 * @file matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 18-04-2021
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "matrix.h"

#define MAX_ROW 20
#define MAX_ROW_PRINT 5
#define MAX_COLUMN 10
#define MAX_COLUMN_PRINT 4

/*******************************/
/*   Helper Functions          */
/*******************************/

/**
 * @brief return random number between 0 ... 1
 *
 * @return double
 */
double randomNumber() { return (double)arc4random() / (double)RAND_MAX; }

/*******************************/
/*         I/O Functions       */
/*******************************/

/**
 * @brief read in DoubleVector data from file
 *
 * @param vec
 * @param filepath
 * @return int ('0' sucessfull)
 */
int readInDoubleVectorData(DoubleVector* vec, const char* filepath) {
  FILE* fp = fopen(filepath, "r");
  if (fp == NULL) {
    return 1;
  }
  int succ_read = 1;
  for (size_t i = 0; i < vec->length; i++) {
    succ_read = fscanf(fp, "%lf", &vec->double_array[i]);
  }
  fclose(fp);

  return succ_read;
}

/**
 * @brief write data form DoubleVector to file
 *
 * @param vec
 * @param filepath
 * @return int
 */
int writeOutDoubleVectorData(DoubleVector* vec, const char* filepath) {
  FILE* fp = fopen(filepath, "w");
  if (fp == NULL) {
    return 1;
  }

  for (size_t i = 0; i < vec->length; i++) {
    if (i < vec->length - 1) {
      fprintf(fp, "%lf\n", vec->double_array[i]);
    } else {
      fprintf(fp, "%lf", vec->double_array[i]);
    }
  }
  fclose(fp);

  return 0;
}

/*******************************/
/*  Double Vector (Dynamic)    */
/*******************************/

/**
 * @brief Create a DoubleVector object (HEAP INI_CAPACITY)
 *
 * @param length
 * @param value
 * @return DoubleVector
 */

DoubleVector* createDoubleVector() {
  DoubleVector* vec = (DoubleVector*)malloc(sizeof(DoubleVector));
  if (!vec) return NULL;

  vec->length = 0u;
  vec->capacity = INIT_CAPACITY;

  double* array = (double*)malloc(vec->capacity * sizeof(double));
  vec->double_array = array;

  return vec;
}

/**
 * @brief Create a Double Vector Of Length object
 *
 * @param length
 * @param value
 * @return DoubleVector*
 */
DoubleVector* createDoubleVectorOfLength(size_t length, double value) {
  DoubleVector* vec = (DoubleVector*)malloc(sizeof(DoubleVector));
  double* array = (double*)malloc(length * sizeof(double));

  for (size_t i = 0; i < length; i++) {
    array[i] = value;
  }

  vec->double_array = array;
  vec->length = length;
  if (vec->length > INIT_CAPACITY) {
    vec->capacity = length + INIT_CAPACITY;
  } else {
    vec->capacity = INIT_CAPACITY;
  }
  return vec;
}

/**
 * @brief Create a Random Double Vector object
 *
 * @param length
 * @return DoubleVector
 */
DoubleVector* createRandomDoubleVectorOfLength(size_t length) {
  DoubleVector* vec = (DoubleVector*)malloc(sizeof(DoubleVector));
  double* array = (double*)malloc(length * sizeof(double));

  for (size_t i = 0; i < length; i++) {
    array[i] = randomNumber();
  }

  vec->double_array = array;
  vec->length = length;
  if (vec->length > INIT_CAPACITY) {
    vec->capacity = length + INIT_CAPACITY;
  } else {
    vec->capacity = INIT_CAPACITY;
  }
  return vec;
}

/**
 * @brief expand allocate memory in HEAP with another INIT_CAPACITY
 *
 * @param vec
 */
void expandDoubleVector(DoubleVector* vec) {
  vec->capacity += vec->capacity;
  vec->double_array =
      realloc(vec->double_array, vec->capacity * sizeof(double));
}

/**
 * @brief shrink allocate memory in HEAP
 *
 * @param vec
 */
void shrinkDoubleVector(DoubleVector* vec) {  // l= 10 // 10
  if ((vec->length >= (vec->capacity - INIT_CAPACITY) &&
       (vec->capacity - INIT_CAPACITY) > 0)) {
    vec->capacity = (vec->capacity - INIT_CAPACITY);

    vec->double_array =
        realloc(vec->double_array, vec->capacity * sizeof(double));
  }
}

/**
 * @brief push (add) new value to vector vec
 *
 * @param vec
 * @param value
 */
void pushValue(DoubleVector* vec, double value) {
  if (vec->length == vec->capacity) {
    expandDoubleVector(vec);
  }
  vec->double_array[vec->length] = value;
  vec->length++;
}

/**
 * @brief pop (get) last element if DoubleVector vec
 *
 * @param vec
 * @return double
 */
double popValue(DoubleVector* vec) {
  double value = vec->double_array[vec->length - 1];

  vec->double_array[vec->length] = 0.0;
  vec->length--;

  if (vec->length < vec->capacity) {
    shrinkDoubleVector(vec);
  }

  return value;
}

/**
 * @brief printf DoubleArray pretty
 *
 * @param DoubleVector* vec
 */
void printDoubleVector(DoubleVector* vec) {
  double* array = vec->double_array;
  size_t length = vec->length;
  for (size_t i = 0; i < length; i++) {
    if (i == 0) {
      printf("[%lf ", array[i]);
    } else if (i == length - 1) {
      printf("%lf]\n", array[i]);
    } else {
      if (length < MAX_COLUMN) {
        printf("%lf ", array[i]);
      } else {
        if (i < MAX_COLUMN_PRINT) {
          printf("%lf ", array[i]);
        } else if (i == MAX_COLUMN_PRINT) {
          printf(" ... ");
        } else if (i > length - MAX_COLUMN_PRINT - 1) {
          printf("%lf ", array[i]);
        }
      }
    }
  }
  printf("Vector 1x%zi, Capacity %zi\n", vec->length, vec->capacity);
}

/**
 * @brief free memory of DoubleVector
 *
 * @param vec
 * @return DoubleVector*
 */
void freeDoubleVector(DoubleVector* vec) {
  free(vec->double_array);
  vec->double_array = NULL;
  free(vec);
  vec = NULL;
}

/**
 * @brief Multiply Vector v1 with Vectot v2  -- dot product!
 * @param col
 * @param row
 * @param length
 * @return double
 */
double multiplyDoubleVectors(DoubleVector* vec1, DoubleVector* vec2) {
  if (vec1->length != vec2->length) {
    perror("vectors have not same length");
    return 0;
  }

  double sum;
  size_t i;
  sum = 0;
  for (i = 0; i < vec1->length; i++) {
    sum += vec1->double_array[i] * vec2->double_array[i];
  }
  return sum;
}

/**
 * @brief add Vector vec1 with Vector vec2
 *
 * @param vec1
 * @param vec2
 * @return DoubleVector*
 */
DoubleVector* addDoubleVector(DoubleVector* vec1, DoubleVector* vec2) {
  if (vec1->length != vec2->length) {
    perror("vectors are not same length");
    return NULL;
  }

  DoubleVector* result = createDoubleVectorOfLength(vec1->length, 0.0);
  for (size_t i = 0; i < vec1->length; i++) {
    result->double_array[i] = vec1->double_array[i] + vec2->double_array[i];
  }

  return result;
}

/**
 * @brief sub Vector vec1 from Vector vec2
 *
 * @param vec1
 * @param vec2
 * @return DoubleVector*
 */
DoubleVector* subDoubleVector(DoubleVector* vec1, DoubleVector* vec2) {
  if (vec1->length != vec2->length) {
    perror("vectors are not same length");
    return NULL;
  }

  DoubleVector* result = createDoubleVectorOfLength(vec1->length, 0.0);
  for (size_t i = 0; i < vec1->length; i++) {
    result->double_array[i] = vec1->double_array[i] - vec2->double_array[i];
  }

  return result;
}

/**
 * @brief multiply each element of Vector vec1 with a scalar
 *
 * @param vec
 * @param scalar
 * @return DoubleVector*
 */
DoubleVector* multiplyScalarToVector(DoubleVector* vec, double scalar) {
  DoubleVector* result = createDoubleVectorOfLength(vec->length, 0.0);
  for (size_t i = 0; i < vec->length; i++) {
    result->double_array[i] = vec->double_array[i] * scalar;
  }
  return result;
}

/**
 * @brief divied each element of Vector vec1 with a scalar
 *
 * @param vec
 * @param scalar
 * @return DoubleVector*
 */
DoubleVector* divideScalarToVector(DoubleVector* vec, double scalar) {
  DoubleVector* result = createDoubleVectorOfLength(vec->length, 0.0);
  for (size_t i = 0; i < vec->length; i++) {
    result->double_array[i] = vec->double_array[i] / scalar;
  }
  return result;
}

/**
 * @brief return mean of Vector vec
 *
 * @param vec
 * @return double
 */
double meanOfDoubleVector(DoubleVector* vec) {
  double mean = 0.0;
  for (size_t i = 0; i < vec->length; i++) {
    mean += vec->double_array[i];
  }

  return (mean / vec->length);
}

/**
 * @brief return min of Vector vec
 *
 * @param vec
 * @return double
 */
double minOfDoubleVector(DoubleVector* vec) {
  double min = vec->double_array[0];
  for (size_t i = 0; i < vec->length; i++) {
    if (min > vec->double_array[i]) min = vec->double_array[i];
  }
  return min;
}

/**
 * @brief return max of Vector vec
 *
 * @param vec
 * @return double
 */
double maxOfDoubleVector(DoubleVector* vec) {
  double max = vec->double_array[0];
  for (size_t i = 0; i < vec->length; i++) {
    if (max < vec->double_array[i]) max = vec->double_array[i];
  }
  return max;
}

/*******************************/
/*        Double Array       */
/*******************************/

/**
 * @brief Create a DoubleArray object in HEAP
 *
 * @param length
 * @param value
 * @return double*
 */
double* createRandomDoubleArray(unsigned int length) {
  double* array = (double*)calloc(length, sizeof(double));

  for (unsigned int i = 0; i < length; i++) {
    array[i] = randomNumber();
  }
  return array;  // die Speicheradresse wird zurückgegeben.
}

/**
 * @brief printf DoubleArray pretty (instead of printVector)
 *
 * @param p_array
 * @param length
 */
void printDoubleArray(double* p_array, unsigned int length) {
  if (length > 1) {
    for (size_t i = 0; i < length; i++) {
      if (i == 0) {
        printf("[%f ", p_array[i]);
      } else if (i == length - 1) {
        printf("%lf]\n", p_array[i]);
      } else {
        if (length < MAX_COLUMN) {
          printf("%f ", p_array[i]);
        } else {
          if (i < MAX_COLUMN_PRINT) {
            printf("%f ", p_array[i]);
          } else if (i == MAX_COLUMN_PRINT) {
            printf(" ... ");
          } else if (i > length - MAX_COLUMN_PRINT - 1) {
            printf("%f ", p_array[i]);
          }
        }
      }
    }
  } else {
    printf("[%lf]\n", p_array[0]);
  }
}

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
  matrix->values[0] = calloc(sizeof(double), 0u);

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
static void expandMatrixRow(DoubleMatrix* mat) {
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
static void expandMatrixColumn(DoubleMatrix* mat) {
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
static void shrinkMatrixColumn(DoubleMatrix* mat) {
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
static void shrinkMatrixRow(DoubleMatrix* mat) {
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
      printDoubleArray(matrix->values[i], matrix->columns);
    }
  } else {
    for (size_t i = 0; i < 4; i++) {
      printDoubleArray(matrix->values[i], matrix->columns);
    }
    printf("...\n");
    for (size_t i = matrix->rows - 4; i < matrix->rows; i++) {
      printDoubleArray(matrix->values[i], matrix->columns);
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
