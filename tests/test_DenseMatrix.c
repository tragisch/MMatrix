#include "dm.h"
#include "dm_internals.h"
#include "dm_io.h"
#include "dm_math.h"
#include "dv_vector.h"
#include "dm_convert.h"
#include <stddef.h>
#include <stdlib.h>

/******************************
 ** Test preconditions:
 *******************************/
#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10

#include "unity.h"
#include "unity_internals.h"

void setUp(void) {
  // set stuff up here
}

void tearDown(void) {
  // clean stuff up here
}

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
 ** Creation of matrices:
 *******************************/

void test_dm_create(void) {
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  size_t cols = 4;
  DoubleMatrix *matrix = dm_create(rows, cols);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->cols);

  for (size_t i = 0; i < matrix->rows; i++) {
    for (size_t j = 0; j < matrix->cols; j++) {
      TEST_ASSERT_EQUAL_DOUBLE(0, dm_get(matrix, i, j));
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

// dm_create_rand
void test_dm_create_rand(void) {

  // Create a random matrix with 2 rows and 3 columns.
  DoubleMatrix *mat = dm_create_rand(2, 3, 0.8);

  // Check that the matrix was created successfully.
  TEST_ASSERT_NOT_NULL(mat);
  TEST_ASSERT_EQUAL_UINT(2, mat->rows);
  TEST_ASSERT_EQUAL_UINT(3, mat->cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      TEST_ASSERT_DOUBLE_WITHIN(1.0, 0.0, dm_get(mat, i, j));
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(mat);
}

void test_dm_create2(void) {
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

// dm_create
void test_dm_create_identity(void) {
  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  DoubleMatrix *matrix = dm_create_identity(rows);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(matrix->rows, matrix->cols);

  for (size_t i = 0; i < matrix->rows; i++) {
    for (size_t j = 0; j < matrix->cols; j++) {
      if (i == j) {
        TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(matrix, i, j));
      } else {
        TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(matrix, i, j));
      }
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

/******************************
 ** Operations on matrices:
 *******************************/

void test_dm_create_from_array() {
  // Test input data
  double array[2][3] = {
      {1.1, 2.2, 3.3},
      {4.4, 5.5, 6.6},
  };
  size_t rows = 2;
  size_t cols = 3;

  // Call the function being tested
  DoubleMatrix *matrix = dm_create_from_array(rows, cols, array);

  // Verify the output
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      TEST_ASSERT_EQUAL_DOUBLE(array[i][j], dm_get(matrix, i, j));
    }
  }

  // Free the input data and output matrix
  dm_destroy(matrix);
}

void test_dm_push_column() {
  // Create a DoubleMatrix to push a column to
  DoubleMatrix *mat = dm_create(3, 2);
  dm_set(mat, 0, 0, 1.0);
  dm_set(mat, 1, 0, 2.0);
  dm_set(mat, 2, 0, 3.0);

  // Create a DoubleVector to push as a column
  DoubleVector *col_vec = dv_vector();
  dv_push_value(col_vec, 4.0);
  dv_push_value(col_vec, 5.0);
  dv_push_value(col_vec, 6.0);

  // Push the column to the DoubleMatrix
  dm_push_column(mat, col_vec);
  // Check if column-capacity is increased correctly:
  TEST_ASSERT_EQUAL(3, mat->cols);

  // Check that the column was pushed correctly
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 2, 1));

  // Clean up
  dm_destroy(mat);
  dv_destroy(col_vec);
}

void test_dv_get_row(void) {
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

void test_dv_get_column(void) {

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

void test_dm_set(void) {
  // Create a matrix with 2 rows and 3 columns.
  DoubleMatrix *mat = dm_create(2, 3);

  // Set the value of the element at row 0, column 1 to 1.23.
  dm_set(mat, 0, 1, 1.23);

  // Check that the value was set correctly.
  TEST_ASSERT_EQUAL_DOUBLE(1.23, dm_get(mat, 0, 1));

  // Free the memory allocated for the matrix.
  dm_destroy(mat);
}

void test_dm_get(void) {
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

void test_dm_get_sub_matrix(void) {
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

  // Verify the values in the sub-matrix
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(sub_mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(sub_mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(sub_mat, 2, 2));

  // Clean up resources
  dm_destroy(mat);
  dm_destroy(sub_mat);
}

void test_dm_create_diagonal(void) {
  // Create a sample diagonal matrix
  size_t rows = 4;
  size_t cols = 4;
  double array[4] = {1.0, 2.0, 3.0, 4.0};
  DoubleMatrix *diagonal_mat = dm_create_diagonal(rows, cols, array);

  // Verify the properties of the diagonal matrix
  TEST_ASSERT_EQUAL(rows, diagonal_mat->rows);
  TEST_ASSERT_EQUAL(cols, diagonal_mat->cols);

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

void test_dm_convert_to_sparse() {
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
  dm_convert(mat, SPARSE);

  // check if matrix is in sparse format
  TEST_ASSERT_EQUAL(SPARSE, mat->format);

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

void test_dm_push_row() {
  // Create a DoubleMatrix to push a row to
  DoubleMatrix *mat = dm_create(3, 2);
  dm_set(mat, 0, 0, 1.0);
  dm_set(mat, 1, 0, 2.0);
  dm_set(mat, 2, 1, 3.0);

  // Create a DoubleVector to push as a row
  DoubleVector *col_row = dv_create(2);
  dv_set(col_row, 0, 4.0);
  dv_set(col_row, 1, 5.0);

  // Push the row to the DoubleMatrix
  dm_push_row(mat, col_row);

  // Check if row-capacity is increased correctly:
  TEST_ASSERT_EQUAL(4, mat->rows);

  // Check that the column was pushed correctly
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 2, 1));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 3, 0));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 3, 1));

  // Clean up
  dm_destroy(mat);
  dv_destroy(col_row);
}

int main(void) {
  UNITY_BEGIN();

  set_default_matrix_format(DENSE);
  RUN_TEST(test_double_precision);
  RUN_TEST(test_dm_create2);
  RUN_TEST(test_dm_create_rand);
  RUN_TEST(test_dm_destroy);
  RUN_TEST(test_dm_get);
  RUN_TEST(test_dm_set);
  RUN_TEST(test_dv_get_column);
  RUN_TEST(test_dv_get_row);
  RUN_TEST(test_dm_get_sub_matrix);
  RUN_TEST(test_dm_create_diagonal);
  RUN_TEST(test_dm_push_column);
  RUN_TEST(test_dm_create_from_array);
  RUN_TEST(test_dm_create_identity);
  RUN_TEST(test_dm_push_row);

  return UNITY_END();
}
