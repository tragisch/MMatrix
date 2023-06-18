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
void test_dm_determinant(matrix_format format) {
  set_default_matrix_format(format);

  double values[2][2] = {{2.0, 3.0}, {1.0, 5.0}};
  DoubleMatrix *mat = dm_create_from_array(2, 2, values);
  double det = dm_determinant(mat);
  TEST_ASSERT_EQUAL_DOUBLE(7.0, det);

  double values2[3][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 10.0}};
  DoubleMatrix *mat2 = dm_create_from_array(3, 3, values2);
  det = dm_determinant(mat2);
  TEST_ASSERT_EQUAL_DOUBLE(-3.0, det);

  // Free memory
  dm_destroy(mat);
  dm_destroy(mat2);
}

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_inverse(matrix_format format) {
  set_default_matrix_format(format);
  // Test case 1: 2x2 matrix
  DoubleMatrix *mat1 = dm_create(2, 2);
  dm_set(mat1, 0, 0, 1);
  dm_set(mat1, 0, 1, 2);
  dm_set(mat1, 1, 0, 3);
  dm_set(mat1, 1, 1, 4);

  DoubleMatrix *inv1 = dm_inverse(mat1);

  TEST_ASSERT_EQUAL_DOUBLE(-2.0, dm_get(inv1, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(inv1, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(1.5, dm_get(inv1, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(-0.5, dm_get(inv1, 1, 1));

  // Test case 2: 3x3 matrix
  DoubleMatrix *mat2 = dm_create(3, 3);
  dm_set(mat2, 0, 0, 1);
  dm_set(mat2, 0, 1, 2);
  dm_set(mat2, 0, 2, 0);
  dm_set(mat2, 1, 0, 2);
  dm_set(mat2, 1, 1, 4);
  dm_set(mat2, 1, 2, 1);
  dm_set(mat2, 2, 0, 2);
  dm_set(mat2, 2, 1, 1);
  dm_set(mat2, 2, 2, 0);

  dm_destroy(mat1);
  dm_destroy(inv1);
}

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_rank_dense(matrix_format format) {
  set_default_matrix_format(format);
  double values[3][3] = {{1.0, 2.0, 3.0}, {0.0, 1.0, 4.0}, {5.0, 6.0, 0.0}};
  DoubleMatrix *mat = dm_create_from_array(3, 3, values);
  size_t rank = dm_rank(mat);
  TEST_ASSERT_EQUAL_INT(3, rank);
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
