#include "dm.h"

#include "dm_convert.h"
#include "dm_internals.h"
#include "dm_io.h"
#include "dm_math.h"
#include "dm_vector.h"
#include <stddef.h>
#include <stdlib.h>

/******************************
 ** Test preconditions:
 *******************************/
#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10

/* Support for Meta Test Rig */
#define TEST_CASE(...)

#include "unity.h"
#include "unity_internals.h"

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

/****************************** /
 ** Test identity matrix creation:
 *******************************/

void test_dm_create_identity_dense(void) {
  set_default_matrix_format(DENSE);
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  DoubleMatrix *matrix = dm_create_identity(rows);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(matrix->rows, matrix->cols);

  for (size_t i = 0; i < matrix->rows; i++) {
    for (size_t j = 0; j < matrix->cols; j++) {
      if (i == j) {
        TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(matrix, i, j));
      } else {
        TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(matrix, i, j));
      }
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

void test_dm_create_identity_csc(void) {
  set_default_matrix_format(CSC);
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  DoubleMatrix *matrix = dm_create_identity(rows);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(matrix->rows, matrix->cols);

  for (size_t i = 0; i < matrix->rows; i++) {
    for (size_t j = 0; j < matrix->cols; j++) {
      if (i == j) {
        TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(matrix, i, j));
      } else {
        TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(matrix, i, j));
      }
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

/******************************
** Test Random Matrix Creation
*******************************/

// dm_create_rand
TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_create_rand(matrix_format format) {
  set_default_matrix_format(format);
  // Create a random matrix with 2 rows and 3 columns.
  DoubleMatrix *mat = dm_create_rand(2, 3, 0.1);

  // Check that the matrix was created successfully.
  TEST_ASSERT_NOT_NULL(mat);
  TEST_ASSERT_NOT_NULL(mat->values);
  TEST_ASSERT_EQUAL_UINT(2, mat->rows);
  TEST_ASSERT_EQUAL_UINT(3, mat->cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      TEST_ASSERT_DOUBLE_WITHIN(1.0, 0.0, dm_get(mat, i, j));
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(mat);
}

/******************************
 ** Test Identity Matrix Creation
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_create_identity(matrix_format format) {
  set_default_matrix_format(format);

  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  DoubleMatrix *matrix = dm_create_identity(rows);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(matrix->rows, matrix->cols);

  for (size_t i = 0; i < matrix->rows; i++) {
    for (size_t j = 0; j < matrix->cols; j++) {
      if (i == j) {
        TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(matrix, i, j));
      } else {
        TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(matrix, i, j));
      }
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

/******************************
 ** Test Array Matrix Creation
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_create_from_array(matrix_format format) {
  set_default_matrix_format(format);

  // Test input data
  double array[2][3] = {
      {1.1, 2.2, 3.3},
      {4.4, 5.5, 6.6},
  };
  size_t rows = 2;
  size_t cols = 3;

  // Call the function being tested
  DoubleMatrix *matrix = dm_create_from_array(rows, cols, array);

  // Verify the output
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      TEST_ASSERT_EQUAL_DOUBLE(array[i][j], dm_get(matrix, i, j));
    }
  }

  // Free the input data and output matrix
  dm_destroy(matrix);
}

/******************************
 ** Test Diagonal Matrix Creation
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_create_diagonal(matrix_format format) {
  set_default_matrix_format(format);
  // Create a sample diagonal matrix
  size_t rows = 4;
  size_t cols = 4;
  double array[4] = {1.0, 2.0, 3.0, 4.0};
  DoubleMatrix *diagonal_mat = dm_create_diagonal(rows, cols, array);

  // Verify the properties of the diagonal matrix
  TEST_ASSERT_EQUAL(rows, diagonal_mat->rows);
  TEST_ASSERT_EQUAL(cols, diagonal_mat->cols);
  TEST_ASSERT_EQUAL(rows, diagonal_mat->nnz);

  // Verify the values in the diagonal matrix
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(diagonal_mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(diagonal_mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(diagonal_mat, 2, 2));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(diagonal_mat, 3, 3));

  // Clean up resources
  dm_destroy(diagonal_mat);
}
