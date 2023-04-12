#include "dm_math.h"
#include "dm_matrix.h"
#include <stddef.h>
#include <stdlib.h>

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10
// #define UPPER_BOUND 100

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Test preconditions:
 *******************************/

enum { INIT_CAPACITY = 2U };

void test_get_row_array() {
  // Test input data
  double arr[3][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}};

  DoubleMatrix *result = dm_create_from_array(3, 3, arr);
  double *row = get_row_array(result, 1);

  TEST_ASSERT_EQUAL_DOUBLE_ARRAY(arr[1], row, 3);

  dm_free_matrix(result);
}

void test_multiply_scalar_matrix(void) {
  // Initialize test data
  DoubleMatrix *mat = dm_create(2, 2);
  mat->values[0][0] = 1.0;
  mat->values[0][1] = 2.0;
  mat->values[1][0] = 3.0;
  mat->values[1][1] = 4.0;

  double scalar = 2.0;

  DoubleMatrix *expected = dm_create(2, 2);

  expected->values[0][0] = 2.0;
  expected->values[0][1] = 4.0;
  expected->values[1][0] = 6.0;
  expected->values[1][1] = 8.0;

  // Call the function to be tested
  multiply_scalar_matrix(mat, scalar);

  // Check the result against the expected
  // output
  TEST_ASSERT_EQUAL_DOUBLE(expected->values[0][0], mat->values[0][0]);
  TEST_ASSERT_EQUAL_DOUBLE(expected->values[0][1], mat->values[0][1]);
  TEST_ASSERT_EQUAL_DOUBLE(expected->values[1][0], mat->values[1][0]);
  TEST_ASSERT_EQUAL_DOUBLE(expected->values[1][1], mat->values[1][1]);
}
