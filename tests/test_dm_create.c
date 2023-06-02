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

/******************************
 ** Test if set_default_matrix_format works
 *******************************/

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_set_default_matrix_format(matrix_format format) {
  set_default_matrix_format(format);
  DoubleMatrix *mat = dm_create(2, 3);
  switch (format) {
  case DENSE:
    TEST_ASSERT_EQUAL(0, mat->format);
    break;
  case COO:
    TEST_ASSERT_EQUAL(1, mat->format);
    break;
  // case CSC:
  //   TEST_ASSERT_EQUAL(2, mat->format);
  //   break;
  default:
    break;
  }
}

/******************************
 ** Creation of matrices:
 *******************************/

void test_dm_create_dense(void) {

  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  size_t cols = 4;
  DoubleMatrix *matrix = dm_create_format(rows, cols, DENSE);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->cols);
  TEST_ASSERT_EQUAL(rows * cols, matrix->capacity);
  TEST_ASSERT_EQUAL(0, matrix->nnz);

  for (size_t i = 0; i < matrix->rows; i++) {
    for (size_t j = 0; j < matrix->cols; j++) {
      TEST_ASSERT_EQUAL_DOUBLE(0, dm_get(matrix, i, j));
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

void test_dm_create_coo(void) {
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  size_t cols = 4;
  DoubleMatrix *matrix = dm_create_format(rows, cols, COO);

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

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_create2(matrix_format format) {
  set_default_matrix_format(format);
  // Create  3x3 double matrix
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

  // test if all values are set correctly
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 1, 2));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(mat, 2, 1));
  TEST_ASSERT_EQUAL_DOUBLE(9.0, dm_get(mat, 2, 2));

  // Free the memory allocated for the matrix.
  dm_destroy(mat);
}

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

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_set(matrix_format format) {
  set_default_matrix_format(format);
  // Create a matrix with 2 rows and 3 columns.
  DoubleMatrix *mat = dm_create(2, 3);

  // Set the value of the element at row 0, column 1 to 1.23.
  dm_set(mat, 0, 1, 1.23);

  // Check that the value was set correctly.
  TEST_ASSERT_EQUAL_DOUBLE(1.23, dm_get(mat, 0, 1));

  // Free the memory allocated for the matrix.
  dm_destroy(mat);
}

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_get(matrix_format format) {
  set_default_matrix_format(format);
  // Create a matrix with 2 rows and 3 columns.
  DoubleMatrix *mat = dm_create(2, 3);

  // Set the value of the element at row 0, column 1 to 1.23.
  dm_set(mat, 0, 1, 1.23);

  // Get the value of the element at row 0, column 1.
  double value = dm_get(mat, 0, 1);

  // Check that the value was retrieved correctly.
  TEST_ASSERT_EQUAL_DOUBLE(1.23, value);

  // Free the memory allocated for the matrix.
  dm_destroy(mat);
}

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

TEST_CASE(0)
TEST_CASE(1)
TEST_CASE(2)
void test_dm_create_diagonal(matrix_format format) {
  set_default_matrix_format(format);
  // Create a sample diagonal matrix
  size_t rows = 4;
  size_t cols = 4;
  double array[4] = {1.0, 2.0, 3.0, 4.0};
  DoubleMatrix *diagonal_mat = dm_create_diagonal(rows, cols, array);

  // Verify the properties of the diagonal matrix
  TEST_ASSERT_EQUAL(rows, diagonal_mat->rows);
  TEST_ASSERT_EQUAL(cols, diagonal_mat->cols);
  TEST_ASSERT_EQUAL(rows, diagonal_mat->nnz);

  // Verify the values in the diagonal matrix
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(diagonal_mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(diagonal_mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(diagonal_mat, 2, 2));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(diagonal_mat, 3, 3));

  // Clean up resources
  dm_destroy(diagonal_mat);
}

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

void test_dm_convert_dense_to_coo() {
  // create dense matrix
  DoubleMatrix *mat = dm_create_format(3, 3, DENSE);
  dm_set(mat, 0, 0, 1);
  dm_set(mat, 0, 1, 2);
  dm_set(mat, 0, 2, 0);
  dm_set(mat, 1, 0, 0);
  dm_set(mat, 1, 1, 3);
  dm_set(mat, 1, 2, 4);
  dm_set(mat, 2, 0, 5);
  dm_set(mat, 2, 1, 0);
  dm_set(mat, 2, 2, 6);

  // convert to sparse matrix
  dm_convert(mat, COO);

  // check if matrix is in sparse format
  TEST_ASSERT_EQUAL(COO, mat->format);

  // check matrix values
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 1, 2));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 2, 2));

  // check number of non-zero elements
  TEST_ASSERT_EQUAL(6, mat->nnz);

  // clean up
  dm_destroy(mat);
}

void test_dm_convert_coo_to_dense() {
  // create a sparse matrix
  DoubleMatrix *sparse_mat = dm_create_format(3, 3, COO);
  dm_set(sparse_mat, 0, 0, 1);
  dm_set(sparse_mat, 0, 1, 0);
  dm_set(sparse_mat, 0, 2, 0);
  dm_set(sparse_mat, 1, 0, 0);
  dm_set(sparse_mat, 1, 1, 2);
  dm_set(sparse_mat, 1, 2, 0);
  dm_set(sparse_mat, 2, 0, 0);
  dm_set(sparse_mat, 2, 1, 0);
  dm_set(sparse_mat, 2, 2, 3);

  // convert to dense matrix
  dm_convert(sparse_mat, DENSE);

  // check if matrix is now in dense format
  TEST_ASSERT_EQUAL(DENSE, sparse_mat->format);

  // check values
  double expected_values[9] = {1, 0, 0, 0, 2, 0, 0, 0, 3};
  for (size_t i = 0; i < 9; i++) {
    TEST_ASSERT_EQUAL_DOUBLE(expected_values[i], sparse_mat->values[i]);
  }

  // destroy matrix
  dm_destroy(sparse_mat);
}
