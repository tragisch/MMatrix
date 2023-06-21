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
 ** SET
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
void test_dm_set(matrix_format format) {
  set_default_matrix_format(format);
  // Create a matrix with 2 rows and 3 columns.
  DoubleMatrix *mat = dm_create(3, 3);

  // Set the value of the element at row 0, column 1 to 1.23.
  dm_set(mat, 0, 0, 0.0);
  dm_set(mat, 0, 1, 1.0);
  dm_set(mat, 0, 2, 2.0);
  dm_set(mat, 1, 0, 3.0);
  dm_set(mat, 1, 1, 4.0);
  dm_set(mat, 1, 2, 5.0);
  dm_set(mat, 2, 0, 6.0);
  dm_set(mat, 2, 1, 7.0);
  dm_set(mat, 2, 2, 8.0);

  // Check that the value was set correctly.
  TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 1, 2));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(mat, 2, 1));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(mat, 2, 2));

  // Free the memory allocated for the matrix.
  dm_destroy(mat);
}

/******************************
 ** GET
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
void test_dm_get(matrix_format format) {
  set_default_matrix_format(format);
  // Create a matrix with 2 rows and 3 columns.
  DoubleMatrix *mat = dm_create(2, 3);

  // Set the value of the element at row 0, column 1 to 1.23.
  dm_set(mat, 0, 1, 1.23);

  // Get the value of the element at row 0, column 1.
  double value = dm_get(mat, 0, 1);

  // Check that the value was retrieved correctly.
  TEST_ASSERT_EQUAL_DOUBLE(1.23, value);

  // Free the memory allocated for the matrix.
  dm_destroy(mat);
}
