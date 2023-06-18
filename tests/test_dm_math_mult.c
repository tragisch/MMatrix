#include "dbg.h"

#include "dm.h"
#include "dm_internals.h"
#include "dm_io.h"
#include "dm_math.h"
#include "dm_vector.h"

#include <stdbool.h>
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
 ** Tests
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_multiply_by_vector(matrix_format format) {
  set_default_matrix_format(format);
  // Create test matrix and vector
  double array[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  DoubleMatrix *matrix = dm_create_from_array(2, 3, array);

  double values[3] = {1.0, 2.0, 3.0};
  DoubleVector *vector = dv_create_from_array(values, 3);

  // Calculate the product of the matrix and vector
  DoubleVector *result = dm_multiply_by_vector(matrix, vector);

  // Create the expected result vector
  double expected_values[2] = {14.0, 32.0};
  DoubleVector *expected_result = dv_create_from_array(expected_values, 2);

  // Compare the result with the expected result
  TEST_ASSERT_TRUE(dv_equal(result, expected_result));

  // Check that the dimensions of the matrix and vector are correct
  TEST_ASSERT_EQUAL_UINT(3, matrix->cols);
  TEST_ASSERT_EQUAL_UINT(2, matrix->rows);
  TEST_ASSERT_EQUAL_UINT(3, vector->rows);

  // Free the memory allocated for the matrix, vector, and result
  dm_destroy(matrix);
  dv_destroy(vector);
  dv_destroy(result);
  dv_destroy(expected_result);
}

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_multiply_by_matrix(matrix_format format) {
  set_default_matrix_format(format);
  // Create two test matrices
  double array1[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  DoubleMatrix *matrix1 = dm_create_from_array(2, 3, array1);

  double array2[3][2] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  DoubleMatrix *matrix2 = dm_create_from_array(3, 2, array2);

  // Calculate the product of the two matrices
  DoubleMatrix *result = dm_multiply_by_matrix(matrix1, matrix2);

  // Create the expected result matrix
  double expected_array[2][2] = {{58.0, 64.0}, {139.0, 154.0}};
  DoubleMatrix *expected_result = dm_create_from_array(2, 2, expected_array);

  // Compare the result with the expected result
  TEST_ASSERT_TRUE(dm_equal(result, expected_result));

  // Free the memory allocated for the matrices and result
  dm_destroy(matrix1);
  dm_destroy(matrix2);
  dm_destroy(result);
  dm_destroy(expected_result);
}

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_multiply_by_scalar(matrix_format format) {
  set_default_matrix_format(format);
  // Create a 2x2 matrix

  DoubleMatrix *mat = dm_create(2, 2);
  dm_set(mat, 0, 0, 1.0);
  dm_set(mat, 0, 1, 2.0);
  dm_set(mat, 1, 0, 3.0);
  dm_set(mat, 1, 1, 4.0);

  // Multiply by a scalar
  dm_multiply_by_scalar(mat, 2.0);

  // Check the result
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(mat, 1, 1));

  // Clean up
  dm_destroy(mat);
}