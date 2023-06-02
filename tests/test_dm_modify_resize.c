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
 ** Resize matrix
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
// TEST_CASE(2)
void test_dm_resize(matrix_format format) {
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

  // Resize the matrix to a new size of 2 rows and 4 columns
  dm_resize(mat, 2, 4);

  // Check the matrix size after resize
  TEST_ASSERT_EQUAL_UINT(2, mat->rows);
  TEST_ASSERT_EQUAL_UINT(4, mat->cols);

  // Check the remaining elements after resize
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(mat, 0, 3)); // New element
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 1, 2));
  TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(mat, 1, 3)); // New element

  // Clean up
  dm_destroy(mat);
}