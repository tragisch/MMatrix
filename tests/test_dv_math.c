#include "dbg.h"
#include "dv_math.h"
#include "dv_vector.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

/******************************
 ** Test preconditions:
 *******************************/

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Tests
 *******************************/

void test_dv_dot_product() {
  // create two vectors to compute the cross product of
  double array1[3] = {1.0, 3.0, -5.0};
  DoubleVector *vec1 = dv_create_from_array(array1, 3);

  double array2[3] = {4.0, -2.0, -1.0};
  DoubleVector *vec2 = dv_create_from_array(array2, 3);

  // compute the cross product
  double cross = dv_dot_product(vec1, vec2);

  // check that the resulting vector has the expected values
  TEST_ASSERT_EQUAL_DOUBLE(3, cross);

  // clean up memory
  dv_destroy(vec1);
  dv_destroy(vec2);
}

void test_dv_add_vector() {
  // create two vectors to add together
  double array1[3] = {1, 2, 3};
  DoubleVector *vec1 = dv_create_from_array(array1, 3);

  double array2[3] = {4, 5, 6};
  DoubleVector *vec2 = dv_create_from_array(array2, 3);

  // add vec2 to vec1
  dv_add_vector(vec1, vec2);

  // check that the resulting vector has the expected values
  TEST_ASSERT_EQUAL_DOUBLE(5, dm_get(vec1, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(7, dm_get(vec1, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(9, dm_get(vec1, 2, 0));

  // clean up memory
  dv_destroy(vec1);
  dv_destroy(vec2);
}

void test_dv_sub_vector() {
  // create two vectors to add together
  double array1[3] = {1, 2, 3};
  DoubleVector *vec1 = dv_create_from_array(array1, 3);

  double array2[3] = {4, 5, 6};
  DoubleVector *vec2 = dv_create_from_array(array2, 3);

  // add vec2 to vec1
  dv_sub_vector(vec1, vec2);

  // check that the resulting vector has the expected values
  TEST_ASSERT_EQUAL_DOUBLE(-4, dm_get(vec1, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(-3, dm_get(vec1, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(-3, dm_get(vec1, 2, 0));

  // clean up memory
  dv_destroy(vec1);
  dv_destroy(vec2);
}

void test_dv_multiply_by_scalar() {
  // create two vectors to add together
  double array1[3] = {1, 2, 3};
  DoubleVector *vec1 = dv_create_from_array(array1, 3);
  double scalar = 4.16;

  // add vec2 to vec1
  dv_multiply_by_scalar(vec1, scalar);

  // check that the resulting vector has the expected values
  TEST_ASSERT_EQUAL_DOUBLE(4.16, dm_get(vec1, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(8.32, dm_get(vec1, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(12.48, dm_get(vec1, 2, 0));

  // clean up memory
  dv_destroy(vec1);
}

void test_dv_divide_by_scalar() {
  // create two vectors to add together
  double array1[3] = {5, 1, 4};
  DoubleVector *vec1 = dv_create_from_array(array1, 3);
  double scalar = 4.16;

  // add vec2 to vec1
  dv_divide_by_scalar(vec1, scalar);

  // check that the resulting vector has the expected values
  TEST_ASSERT_EQUAL_DOUBLE(1.20192, dm_get(vec1, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(0.240385, dm_get(vec1, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(0.961538, dm_get(vec1, 2, 0));

  // clean up memory
  dv_destroy(vec1);
}

void test_dv_add_constant() {
  // create two vectors to add together
  double array1[3] = {5, 1, 4};
  DoubleVector *vec1 = dv_create_from_array(array1, 3);
  double scalar = 2.445;

  // add constant to vec
  dv_add_constant(vec1, scalar);

  // check that the resulting vector has the expected values
  TEST_ASSERT_EQUAL_DOUBLE(7.445, dm_get(vec1, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(3.445, dm_get(vec1, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.445, dm_get(vec1, 2, 0));

  // clean up memory
  dv_destroy(vec1);
}

void test_dv_mean() {
  // create two vectors to add together
  double array1[3] = {5, 1, 4};
  DoubleVector *vec1 = dv_create_from_array(array1, 3);

  // mean of vec1
  double mean = dv_mean(vec1);

  // check that the resulting vector has the expected values
  TEST_ASSERT_EQUAL_DOUBLE(3.33333, mean);

  // clean up memory
  dv_destroy(vec1);
}

void test_dv_min() {
  // create two vectors to add together
  double array1[3] = {5, 1, 4};
  DoubleVector *vec1 = dv_create_from_array(array1, 3);

  // min of vec1
  double min = dv_min(vec1);

  // check that the resulting vector has the expected values
  TEST_ASSERT_EQUAL_DOUBLE(1, min);

  // clean up memory
  dv_destroy(vec1);
}

void test_dv_max() {
  // create two vectors to add together
  double array1[3] = {5, 1, 4};
  DoubleVector *vec1 = dv_create_from_array(array1, 3);

  // min of vec1
  double max = dv_max(vec1);

  // check that the resulting vector has the expected values
  TEST_ASSERT_EQUAL_DOUBLE(5, max);

  // clean up memory
  dv_destroy(vec1);
}

void test_dv_magnitude() {
  DoubleVector *vec = dv_create(3);
  dv_set(vec, 0, 1.0);
  dv_set(vec, 1, 2.0);
  dv_set(vec, 2, 2.0);

  double expected = 3.0;
  double result = dv_magnitude(vec);

  TEST_ASSERT_EQUAL_FLOAT(expected, result);

  dv_destroy(vec);
}

void test_dv_normalize() {
  double data[3] = {1.0, 2.0, 3.0};
  DoubleVector *vec = dv_create_from_array(data, 3);
  dv_normalize(vec);
  TEST_ASSERT_EQUAL_DOUBLE(0.267261, dv_get(vec, 0));
  TEST_ASSERT_EQUAL_DOUBLE(0.534522, dv_get(vec, 1));
  TEST_ASSERT_EQUAL_DOUBLE(0.801783, dv_get(vec, 2));

  // free memory
  dv_destroy(vec);
}
