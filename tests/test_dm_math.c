#include "dbg.h"
#include "dm_math.h"
#include "dm_matrix.h"
#include <stdbool.h>
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

void test_dm_get_row_as_array() {
  // Test input data
  double arr[3][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}};

  DoubleMatrix *result = dm_create_from_array(3, 3, arr);
  double *row = dm_get_row_as_array(result, 1);

  TEST_ASSERT_EQUAL_DOUBLE_ARRAY(arr[1], row, 3);

  dm_destroy(result);
}

// void test_dm_multiply_by_scalar(void) {
//   // Initialize test data
//   DoubleMatrix *mat = dm_create(2, 2);
//   dm_set(mat, 0, 0, 1.0);
//   dm_set(mat, 0, 1, 2.0);
//   dm_set(mat, 1, 0, 3.0);
//   dm_set(mat, 1, 1, 4.0);

//   double scalar = 2.0;

//   DoubleMatrix *expected = dm_create(2, 2);
//   dm_set(mat, 0, 0, 2.0);
//   dm_set(mat, 0, 1, 4.0);
//   dm_set(mat, 1, 0, 6.0);
//   dm_set(mat, 1, 1, 8.0);

//   // Call the function to be tested
//   dm_multiply_by_scalar(mat, scalar);

//   // Check the result against the expected
//   // output
//   TEST_ASSERT_EQUAL_DOUBLE(dm_get(expected, 0, 0), dm_get(mat, 0, 0));
//   TEST_ASSERT_EQUAL_DOUBLE(dm_get(expected, 0, 1), dm_get(mat, 0, 1));
//   TEST_ASSERT_EQUAL_DOUBLE(dm_get(expected, 1, 0), dm_get(mat, 1, 0));
//   TEST_ASSERT_EQUAL_DOUBLE(dm_get(expected, 1, 1), dm_get(mat, 1, 1));

//   // Free the memory allocated for the matrix
//   dm_destroy(mat);
//   dm_destroy(expected);
// }

void test_dm_transpose() {
  // Create a test matrix
  double arr[4][4] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  DoubleMatrix *matrix = dm_create_from_array(3, 3, arr);

  // Transpose the matrix
  dm_transpose(matrix);

  // Check that the matrix has the expected dimensions and values
  TEST_ASSERT_EQUAL(3, matrix->rows);
  TEST_ASSERT_EQUAL(3, matrix->cols);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(matrix, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(matrix, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(matrix, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(matrix, 0, 3));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(matrix, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(matrix, 1, 2));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(matrix, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(matrix, 2, 1));
  TEST_ASSERT_EQUAL_DOUBLE(9.0, dm_get(matrix, 2, 2));

  // Free the memory allocated for the matrix
  dm_destroy(matrix);
}

// void test_dm_equal_matrix() {
//   // Create two test matrices with equal dimensions and values
//   double array1[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
//   DoubleMatrix *matrix1 = dm_create_from_array(2, 3, array1);
//   DoubleMatrix *matrix2 = dm_create_from_array(2, 3, array1);

//   // Compare the matrices and ensure they are equal
//   TEST_ASSERT_TRUE(dm_equal_matrix(matrix1, matrix2));

//   // Modify one of the matrices and ensure they are no longer equal
//   matrix1->values[0][0] = 7.0;
//   TEST_ASSERT_FALSE(dm_equal_matrix(matrix1, matrix2));

//   // Free the memory allocated for the matrices
//   dm_destroy(matrix1);
//   dm_destroy(matrix2);
// }

void test_dm_multiply_with_matrix() {
  // Create two test matrices
  double array1[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  DoubleMatrix *matrix1 = dm_create_from_array(2, 3, array1);

  double array2[3][2] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  DoubleMatrix *matrix2 = dm_create_from_array(3, 2, array2);

  // Calculate the product of the two matrices
  DoubleMatrix *result = dm_multiply_with_matrix(matrix1, matrix2);

  // Create the expected result matrix
  double expected_array[2][2] = {{58.0, 64.0}, {139.0, 154.0}};
  DoubleMatrix *expected_result = dm_create_from_array(2, 2, expected_array);

  // Compare the result with the expected result
  TEST_ASSERT_TRUE(dm_equal_matrix(result, expected_result));

  // Free the memory allocated for the matrices and result
  dm_destroy(matrix1);
  dm_destroy(matrix2);
  dm_destroy(result);
  dm_destroy(expected_result);
}

void test_dv_multiply_with_matrix() {
  // Create test matrix and vector
  double array[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  DoubleMatrix *matrix = dm_create_from_array(2, 3, array);

  double values[3] = {1.0, 2.0, 3.0};
  DoubleVector *vector = dv_create_from_array(values, 3);

  // Calculate the product of the matrix and vector
  DoubleVector *result = dv_multiply_with_matrix(vector, matrix);

  // Create the expected result vector
  double expected_values[2] = {14.0, 32.0};
  DoubleVector *expected_result = dv_create_from_array(expected_values, 2);

  // Compare the result with the expected result
  TEST_ASSERT_TRUE(dv_equal(result, expected_result));

  // Check that the dimensions of the matrix and vector are correct
  TEST_ASSERT_EQUAL_UINT(3, matrix->cols);
  TEST_ASSERT_EQUAL_UINT(2, matrix->rows);
  TEST_ASSERT_EQUAL_UINT(3, vector->rows);

  // Free the memory allocated for the matrix, vector, and result
  dm_destroy(matrix);
  dv_destroy(vector);
  dv_destroy(result);
  dv_destroy(expected_result);
}

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