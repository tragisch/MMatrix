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
void test_dm_transpose(matrix_format format) {
  set_default_matrix_format(format);
  // Create a test matrix
  double arr[3][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  DoubleMatrix *matrix = dm_create_from_array(3, 3, arr);

  // Transpose the matrix
  dm_transpose(matrix);

  // Check that the matrix has the expected dimensions and values
  TEST_ASSERT_EQUAL(3, matrix->rows);
  TEST_ASSERT_EQUAL(3, matrix->cols);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(matrix, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(matrix, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(matrix, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(matrix, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(matrix, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(matrix, 1, 2));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(matrix, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(matrix, 2, 1));
  TEST_ASSERT_EQUAL_DOUBLE(9.0, dm_get(matrix, 2, 2));

  // Free the memory allocated for the matrix
  dm_destroy(matrix);
}

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_trace(matrix_format format) {
  set_default_matrix_format(format);
  // Create a 3x3 matrix
  DoubleMatrix *mat = dm_create(3, 3);
  dm_set(mat, 0, 0, 1.0);
  dm_set(mat, 0, 1, 2.0);
  dm_set(mat, 0, 2, 3.0);
  dm_set(mat, 1, 0, 4.0);
  dm_set(mat, 1, 1, 5.0);
  dm_set(mat, 1, 2, 6.0);
  dm_set(mat, 2, 0, 7.0);
  dm_set(mat, 2, 1, 8.0);
  dm_set(mat, 2, 2, 9.0);

  // Calculate the trace
  double trace = dm_trace(mat);

  // Check the result
  TEST_ASSERT_EQUAL_DOUBLE(15.0, trace);

  // Clean up
  dm_destroy(mat);
}
