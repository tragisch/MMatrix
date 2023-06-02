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
 ** Get Row
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_get_row(matrix_format format) {
  set_default_matrix_format(format);
  // create test matrix
  double values[3][4] = {
      {1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.9, 10.0, 11.0, 12.0}};
  DoubleMatrix *mat = dm_create_from_array(3, 4, values);

  // get row vector
  DoubleVector *vec = dm_get_row(mat, 1);

  // check vector length
  TEST_ASSERT_EQUAL_INT(4, vec->rows);

  // check vector values
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dv_get(vec, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dv_get(vec, 1));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dv_get(vec, 2));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dv_get(vec, 3));

  // free memory
  dm_destroy(mat);
  dv_destroy(vec);
}

/******************************
 ** Get Column
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_get_column(matrix_format format) {
  set_default_matrix_format(format);
  double values[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  DoubleMatrix *mat = dm_create_from_array(3, 2, values);
  DoubleVector *vec = dm_get_column(mat, 1);
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dv_get(vec, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dv_get(vec, 1));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dv_get(vec, 2));
  TEST_ASSERT_EQUAL(3, vec->rows);
  dm_destroy(mat);
  dv_destroy(vec);
}

/******************************
 ** Get Sub Matrix
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_get_sub_matrix(matrix_format format) {
  set_default_matrix_format(format);
  // Create a sample matrix
  DoubleMatrix *mat = dm_create(5, 5);

  // Set values in the matrix
  dm_set(mat, 0, 0, 1.0);
  dm_set(mat, 0, 1, 2.0);
  dm_set(mat, 1, 1, 3.0);
  dm_set(mat, 2, 2, 4.0);
  dm_set(mat, 3, 3, 5.0);
  dm_set(mat, 4, 4, 6.0);

  // Get the sub-matrix
  DoubleMatrix *sub_mat = dm_get_sub_matrix(mat, 1, 3, 1, 3);

  // Verify the properties of the sub-matrix
  TEST_ASSERT_EQUAL(3, sub_mat->rows);
  TEST_ASSERT_EQUAL(3, sub_mat->cols);
  TEST_ASSERT_EQUAL(3, sub_mat->nnz);

  // Verify the values in the sub-matrix
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(sub_mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(sub_mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(sub_mat, 2, 2));

  // Clean up resources
  dm_destroy(mat);
  dm_destroy(sub_mat);
}
