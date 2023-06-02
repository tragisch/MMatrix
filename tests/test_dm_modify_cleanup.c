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

#define EPSILON 1e-10

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Drop small entries
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
// TEST_CASE(2)
void test_dm_drop_small_entries(matrix_format format) {
  set_default_matrix_format(format);
  // Create a matrix with some initial values
  DoubleMatrix *mat = dm_create(3, 3);
  dm_set(mat, 0, 0, 1.0);
  dm_set(mat, 0, 1, 0.5);
  dm_set(mat, 0, 2, 0.000001);
  dm_set(mat, 1, 0, 0.0);
  dm_set(mat, 1, 1, 1.5);
  dm_set(mat, 1, 2, 0.00000000002);
  dm_set(mat, 2, 0, 0.00000000003);
  dm_set(mat, 2, 1, 0.00000000004);
  dm_set(mat, 2, 2, 2.0);

  // Drop small entries from the matrix
  dm_drop_small_entries(mat);

  // Check the remaining values
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(0.5, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(0.000001, dm_get(mat, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(1.5, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(mat, 1, 2));
  TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(mat, 2, 1));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 2, 2));

  // Clean up
  dm_destroy(mat);
}
