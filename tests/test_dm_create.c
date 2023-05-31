#include "dm.h"
#include <stddef.h>
#include <stdlib.h>

/****************************** /
 ** Test preconditions:
 *******************************/

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 100

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Test if precision works
 *******************************/

void test_double_precision(void) {
  double value1 = 1.23456789;
  double value2 = 1.23456788;

  TEST_ASSERT_EQUAL_DOUBLE(value1, value2);
  // This assertion will fail because value1 and value2 are not exactly equal.

  TEST_ASSERT_EQUAL_DOUBLE(value1, value2 + 0.00000001);
  // This assertion will pass because value2 + 0.00000001 is within the
  // precision specified by UNITY_DOUBLE_PRECISION.
}

/****************************** /
 ** Simple constructor tests:
 *******************************/

void test_dm_create_dense(void) {
  set_default_matrix_format(DENSE);
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  size_t cols = 4;
  DoubleMatrix *matrix = dm_create(rows, cols);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->cols);
  TEST_ASSERT_EQUAL(0, matrix->nnz);
  TEST_ASSERT_EQUAL(rows * cols, matrix->capacity);
  TEST_ASSERT_EQUAL(0, matrix->format);

  for (size_t i = 0; i < matrix->rows; i++) {
    for (size_t j = 0; j < matrix->cols; j++) {
      double val = matrix->values[i * matrix->cols + j];
      TEST_ASSERT_EQUAL_DOUBLE(0, val);
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

void test_dm_create_coo(void) {
  set_default_matrix_format(COO);
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  size_t cols = 4;
  DoubleMatrix *matrix = dm_create(rows, cols);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->cols);
  TEST_ASSERT_EQUAL(0, matrix->nnz);
  TEST_ASSERT_NULL(matrix->col_ptr);
  TEST_ASSERT_EQUAL(INIT_CAPACITY, matrix->capacity);
  TEST_ASSERT_EQUAL(1, matrix->format);

  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

void test_dm_create_csc(void) {
  set_default_matrix_format(CSC);
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  size_t cols = 4;
  DoubleMatrix *matrix = dm_create(rows, cols);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->cols);
  TEST_ASSERT_EQUAL(0, matrix->nnz);
  TEST_ASSERT_NULL(matrix->col_indices);
  TEST_ASSERT_EQUAL(INIT_CAPACITY, matrix->capacity);
  TEST_ASSERT_EQUAL(2, matrix->format);

  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

/****************************** /
 ** Simple deconstructor tests:
 *******************************/

void test_dm_destroy() {
  size_t rows = 10;
  size_t cols = 10;
  double density = 0.5;
  DoubleMatrix *sp_matrix = dm_create_rand(rows, cols, density);

  // Call the function under test
  dm_destroy(sp_matrix);

  // Check that all memory was freed
  TEST_ASSERT_NULL(sp_matrix->row_indices);
  TEST_ASSERT_NULL(sp_matrix->col_indices);
  TEST_ASSERT_NULL(sp_matrix->values);
}
