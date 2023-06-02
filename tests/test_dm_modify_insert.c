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
 ** Insert Column
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_insert_column(matrix_format format) {
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

  // Create a vector to insert
  DoubleVector *vec = dv_create(3);
  dv_set(mat, 0, 10.0);
  dv_set(mat, 1, 11.0);
  dv_set(mat, 2, 12.0);

  // Insert the vector at column index 1
  dm_insert_column(mat, 1, vec);

  // Check the matrix values after insertion
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(10.0, dm_get(mat, 0, 1)); // Inserted vector values
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(11.0, dm_get(mat, 1, 1)); // Inserted vector values
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 1, 2));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(12.0, dm_get(mat, 2, 1)); // Inserted vector values
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(mat, 2, 2));

  // Clean up
  dm_destroy(mat);
  dv_destroy(vec);
}

/******************************
 ** Insert Row
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_insert_row(matrix_format format) {
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

  // Create a vector to insert
  DoubleVector *vec = dv_create(3);
  dv_set(vec, 0, 10.0);
  dv_set(vec, 1, 11.0);
  dv_set(vec, 2, 12.0);

  // Insert the vector at row index 1
  dm_insert_row(mat, 1, vec);

  // Check the matrix values after insertion
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(10.0, dm_get(mat, 1, 0)); // Inserted vector values
  TEST_ASSERT_EQUAL_DOUBLE(11.0, dm_get(mat, 1, 1)); // Inserted vector values
  TEST_ASSERT_EQUAL_DOUBLE(12.0, dm_get(mat, 1, 2)); // Inserted vector values
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 2, 1));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 2, 2));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(mat, 3, 0));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(mat, 3, 1));
  TEST_ASSERT_EQUAL_DOUBLE(9.0, dm_get(mat, 3, 2));

  // Clean up
  dm_destroy(mat);
  dv_destroy(vec);
}
