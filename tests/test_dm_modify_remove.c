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
 ** Remove sparse element
 *******************************/

TEST_CASE(1)
TEST_CASE(2)
void test_dm_remove_entry(matrix_format format) {
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

  // Remove the entry at row 1, column 1
  dm_remove_entry(mat, 1, 1);

  // Check the matrix values after removal
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(mat, 1, 1)); // Entry removed
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(mat, 2, 1));
  TEST_ASSERT_EQUAL_DOUBLE(9.0, dm_get(mat, 2, 2));

  // Clean up
  dm_destroy(mat);
}

/******************************
 ** Remove column
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_remove_column(matrix_format format) {
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

  // Remove column at index 1
  dm_remove_column(mat, 1);

  // Check the matrix values after removal
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(9.0, dm_get(mat, 2, 1));

  // Check if size is correct
  TEST_ASSERT_EQUAL(3, mat->rows);
  TEST_ASSERT_EQUAL(2, mat->cols);

  // Clean up
  dm_destroy(mat);
}

/******************************
 ** Remove row
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_remove_row(matrix_format format) {
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

  // Remove row at index 1
  dm_remove_row(mat, 1);

  // Check the matrix values after removal
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(9.0, dm_get(mat, 1, 2));

  // Check if size is correct
  TEST_ASSERT_EQUAL(2, mat->rows);
  TEST_ASSERT_EQUAL(3, mat->cols);

  // Clean up
  dm_destroy(mat);
}
