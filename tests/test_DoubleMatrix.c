#include "dm_math.h"
#include "dm_matrix.h"
#include <stddef.h>
#include <stdlib.h>

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10
// #define UPPER_BOUND 100

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Test preconditions:
 *******************************/

enum { INIT_CAPACITY = 2U };

/******************************
 ** Test if precision works
 *******************************/

void test_double_precision(void) {
  double value1 = 1.23456789;
  double value2 = 1.23456788;

  TEST_ASSERT_EQUAL_DOUBLE(value1, value2);
  // This assertion will fail because value1 and value2 are not exactly equal.

  TEST_ASSERT_EQUAL_DOUBLE(value1, value2 + 0.00000001);
  // This assertion will pass because value2 + 0.00000001 is within the
  // precision specified by UNITY_DOUBLE_PRECISION.
}

/******************************
 ** Creation of matrices:
 *******************************/

// create_dm_matrix
void test_create_dm_matrix(void) {
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  size_t cols = 4;
  DoubleMatrix *matrix = create_dm_matrix(rows, cols);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->columns);
  TEST_ASSERT(INIT_CAPACITY <= matrix->rowCapacity);
  TEST_ASSERT(INIT_CAPACITY <= matrix->columnCapacity);

  for (size_t i = 0; i < matrix->rowCapacity; i++) {
    for (size_t j = 0; j < matrix->columnCapacity; j++) {
      TEST_ASSERT_EQUAL_DOUBLE(0.0, matrix->values[i][j]);
    }
  }
  // Free the memory allocated for the matrix.
  free_dm_matrix(matrix);
}

// test new_dm_matrix
void test_matrix_create(void) {
  // Create a new matrix:
  DoubleMatrix *mat = new_dm_matrix();

  // Check that the matrix is created successfully.
  TEST_ASSERT_NOT_NULL(mat);
  TEST_ASSERT_NOT_NULL(mat->values);
  TEST_ASSERT_EQUAL_UINT(INIT_CAPACITY, mat->rowCapacity);
  TEST_ASSERT_EQUAL_UINT(INIT_CAPACITY, mat->columnCapacity);

  free_dm_matrix(mat);
}

// create_rand_dm_matrix
void test_create_rand_dm_matrix(void) {

  // Create a random matrix with 2 rows and 3 columns.
  DoubleMatrix *mat = create_rand_dm_matrix(2, 3);

  // Check that the matrix was created successfully.
  TEST_ASSERT_NOT_NULL(mat);
  TEST_ASSERT_NOT_NULL(mat->values);
  TEST_ASSERT_EQUAL_UINT(2, mat->rows);
  TEST_ASSERT_EQUAL_UINT(3, mat->columns);
  TEST_ASSERT(INIT_CAPACITY <= mat->rowCapacity);
  TEST_ASSERT(INIT_CAPACITY <= mat->columnCapacity);

  for (size_t i = 0; i < mat->rowCapacity; i++) {
    for (size_t j = 0; j < mat->columnCapacity; j++) {
      TEST_ASSERT_DOUBLE_WITHIN(1.0, 0.0, mat->values[i][j]);
    }
  }
  // Free the memory allocated for the matrix.
  free_dm_matrix(mat);
}

// create_dm_matrix
void test_create_identity_matrix(void) {
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  DoubleMatrix *matrix = create_identity_matrix(rows);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(matrix->rows, matrix->columns);
  TEST_ASSERT(INIT_CAPACITY <= matrix->rowCapacity);

  for (size_t i = 0; i < matrix->rowCapacity; i++) {
    for (size_t j = 0; j < matrix->columnCapacity; j++) {
      if (i == j) {
        TEST_ASSERT_EQUAL_DOUBLE(1.0, matrix->values[i][j]);
      } else {
        TEST_ASSERT_EQUAL_DOUBLE(0.0, matrix->values[i][j]);
      }
    }
  }
  // Free the memory allocated for the matrix.
  free_dm_matrix(matrix);
}

/******************************
 ** Operations on matrices:
 *******************************/

void test_set_array_to_dm_matrix() {
  // Test input data
  size_t rows = 2;
  size_t cols = 3;
  double **array = (double **)malloc(rows * sizeof(double *));
  for (size_t i = 0; i < rows; i++) {
    array[i] = (double *)malloc(cols * sizeof(double));
    for (size_t j = 0; j < cols; j++) {
      array[i][j] = i + j;
    }
  }

  // Call the function being tested
  DoubleMatrix *matrix = set_array_to_dm_matrix(rows, cols, array);

  // Verify the output
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->columns);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      TEST_ASSERT_EQUAL_DOUBLE(array[i][j], matrix->values[i][j]);
    }
  }

  // Free the input data and output matrix
  for (size_t i = 0; i < rows; i++) {
    free(array[i]);
  }
  free(array);
  free_dm_matrix(matrix);
}

void test_push_column() {
  // Create a DoubleMatrix to push a column to
  DoubleMatrix *mat = create_dm_matrix(3, 2);
  mat->values[0][0] = 1.0;
  mat->values[1][0] = 2.0;
  mat->values[2][0] = 3.0;

  // Create a DoubleVector to push as a column
  DoubleVector *col_vec = new_dm_vector();
  push_value(col_vec, 4.0);
  push_value(col_vec, 5.0);
  push_value(col_vec, 6.0);

  // Push the column to the DoubleMatrix
  push_column(mat, col_vec);

  // Check if column-capacity is increased correctly:
  TEST_ASSERT(mat->columns <= mat->columnCapacity);
  TEST_ASSERT_EQUAL(5, mat->rowCapacity);

  // Check that the column was pushed correctly
  TEST_ASSERT_EQUAL_DOUBLE(1.0, mat->values[0][0]);
  TEST_ASSERT_EQUAL_DOUBLE(2.0, mat->values[1][0]);
  TEST_ASSERT_EQUAL_DOUBLE(3.0, mat->values[2][0]);
  TEST_ASSERT_EQUAL_DOUBLE(4.0, mat->values[0][1]);
  TEST_ASSERT_EQUAL_DOUBLE(5.0, mat->values[1][1]);
  TEST_ASSERT_EQUAL_DOUBLE(6.0, mat->values[2][1]);

  // Clean up
  free_dm_matrix(mat);
  free_dm_vector(col_vec);
}

void test_push_row() {
  // Create a DoubleMatrix to push a row to
  DoubleMatrix *mat = create_dm_matrix(3, 2);
  mat->values[0][0] = 1.0;
  mat->values[1][0] = 2.0;
  mat->values[2][0] = 3.0;

  // Create a DoubleVector to push as a row
  DoubleVector *col_row = new_dm_vector();
  push_value(col_row, 4.0);
  push_value(col_row, 5.0);
  push_value(col_row, 6.0);

  // Push the column to the DoubleMatrix
  push_row(mat, col_row);

  // Check if column-capacity is increased correctly:
  TEST_ASSERT(mat->rows <= mat->rowCapacity);
  TEST_ASSERT_EQUAL(2, mat->columnCapacity);

  // Check that the column was pushed correctly
  TEST_ASSERT_EQUAL_DOUBLE(1.0, mat->values[0][0]);
  TEST_ASSERT_EQUAL_DOUBLE(2.0, mat->values[1][0]);
  TEST_ASSERT_EQUAL_DOUBLE(3.0, mat->values[2][0]);
  TEST_ASSERT_EQUAL_DOUBLE(4.0, mat->values[3][0]);
  TEST_ASSERT_EQUAL_DOUBLE(5.0, mat->values[3][1]);

  // Clean up
  free_dm_matrix(mat);
  free_dm_vector(col_row);
}

void test_get_row_vector(void) {
  // create test matrix
  DoubleMatrix *mat = create_dm_matrix(3, 4);
  double **values = mat->values;
  values[0][0] = 1.0;
  values[0][1] = 2.0;
  values[0][2] = 3.0;
  values[0][3] = 4.0;
  values[1][0] = 5.0;
  values[1][1] = 6.0;
  values[1][2] = 7.0;
  values[1][3] = 8.0;
  values[2][0] = 9.0;
  values[2][1] = 10.0;
  values[2][2] = 11.0;
  values[2][3] = 12.0;

  // get row vector
  DoubleVector *vec = get_row_vector(mat, 1);

  // check vector length
  TEST_ASSERT_EQUAL_INT(4, vec->length);

  // check vector values
  TEST_ASSERT_EQUAL_DOUBLE(5.0, vec->mat1D->values[0][0]);
  TEST_ASSERT_EQUAL_DOUBLE(6.0, vec->mat1D->values[1][0]);
  TEST_ASSERT_EQUAL_DOUBLE(7.0, vec->mat1D->values[2][0]);
  TEST_ASSERT_EQUAL_DOUBLE(8.0, vec->mat1D->values[3][0]);

  // free memory
  free_dm_matrix(mat);
  free_dm_vector(vec);
}

void test_get_column_vector(void) {
  DoubleMatrix *mat = create_dm_matrix(3, 2);
  double values[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->columns; j++) {
      mat->values[i][j] = values[i][j];
    }
  }
  DoubleVector *vec = get_column_vector(mat, 1);
  TEST_ASSERT_EQUAL_DOUBLE(2.0, vec->mat1D->values[0][0]);
  TEST_ASSERT_EQUAL_DOUBLE(4.0, vec->mat1D->values[1][0]);
  TEST_ASSERT_EQUAL_DOUBLE(6.0, vec->mat1D->values[2][0]);
  TEST_ASSERT_EQUAL(3, vec->length);
  free_dm_matrix(mat);
  free_dm_vector(vec);
}

void test_get_row_array() {
  DoubleMatrix *mat = create_dm_matrix(3, 3);
  double **arr = (double **)malloc(3 * sizeof(double *));
  for (size_t i = 0; i < 3; i++) {
    arr[i] = (double *)malloc(3 * sizeof(double));
    for (size_t j = 0; j < 3; j++) {
      arr[i][j] = i + j;
      mat->values[i][j] = i + j;
    }
  }

  DoubleMatrix *result = set_array_to_dm_matrix(3, 3, arr);
  double *row = get_row_array(result, 1);

  TEST_ASSERT_EQUAL_DOUBLE_ARRAY(arr[1], row, 3);

  free_dm_matrix(mat);
  free(arr);
  free_dm_matrix(result);
}

void test_multiply_scalar_matrix(void) {
  // Initialize test data
  DoubleMatrix *mat = create_dm_matrix(2, 2);
  mat->values[0][0] = 1.0;
  mat->values[0][1] = 2.0;
  mat->values[1][0] = 3.0;
  mat->values[1][1] = 4.0;

  double scalar = 2.0;

  DoubleMatrix *expected = create_dm_matrix(2, 2);

  expected->values[0][0] = 2.0;
  expected->values[0][1] = 4.0;
  expected->values[1][0] = 6.0;
  expected->values[1][1] = 8.0;

  // Call the function to be tested
  multiply_scalar_matrix(mat, scalar);

  // Check the result against the expected
  // output
  TEST_ASSERT_EQUAL_DOUBLE(expected->values[0][0], mat->values[0][0]);
  TEST_ASSERT_EQUAL_DOUBLE(expected->values[0][1], mat->values[0][1]);
  TEST_ASSERT_EQUAL_DOUBLE(expected->values[1][0], mat->values[1][0]);
  TEST_ASSERT_EQUAL_DOUBLE(expected->values[1][1], mat->values[1][1]);
}
