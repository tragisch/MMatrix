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
 ** Reshape matrix
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
// TEST_CASE(2)
void test_dm_reshape(matrix_format format) {
  set_default_matrix_format(format);
  // Create a matrix with some initial values
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

  // Reshape the matrix to a new shape of 1 row and 9 columns
  dm_reshape(mat, 1, 9);

  // Check the matrix shape after reshape
  TEST_ASSERT_EQUAL_UINT(1, mat->rows);
  TEST_ASSERT_EQUAL_UINT(9, mat->cols);

  // Check the values after reshape
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 0, 3));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 0, 4));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 0, 5));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(mat, 0, 6));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(mat, 0, 7));
  TEST_ASSERT_EQUAL_DOUBLE(9.0, dm_get(mat, 0, 8));

  // Clean up
  dm_destroy(mat);
}