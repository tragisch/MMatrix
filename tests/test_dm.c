#include "dm.h"

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
  TEST_ASSERT_EQUAL(rows * cols, matrix->capacity);

  for (size_t i = 0; i < matrix->rows; i++) {
    for (size_t j = 0; j < matrix->cols; j++) {
      TEST_ASSERT_EQUAL_DOUBLE(0, dm_get(matrix, i, j));
    }
  }
  // Free the memory allocated for the matrix.
  dm_destroy(matrix);
}

void test_dm_convert_array(void) {
  double values[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  DoubleMatrix *mat = dm_convert_array(3, 2, values);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 2, 1));
  dm_destroy(mat);
}

void test_dm_set(void) {
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

void test_dm_get(void) {
  double values[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  DoubleMatrix *mat = dm_convert_array(3, 2, values);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(mat, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(3.0, dm_get(mat, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(mat, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(mat, 2, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(mat, 2, 1));
  dm_destroy(mat);
}

void test_dm_get_row() {
  // create test matrix
  double values[3][4] = {
      {1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.9, 10.0, 11.0, 12.0}};
  DoubleMatrix *mat = dm_convert_array(3, 4, values);

  // get row vector
  DoubleMatrix *vec = dm_get_row(mat, 1);

  // check vector length
  TEST_ASSERT_EQUAL_INT(1, vec->rows);

  // check vector values
  TEST_ASSERT_EQUAL_DOUBLE(5.0, dm_get(vec, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(vec, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(7.0, dm_get(vec, 1, 2));
  TEST_ASSERT_EQUAL_DOUBLE(8.0, dm_get(vec, 1, 3));

  // free memory
  dm_destroy(mat);
  dm_destroy(vec);
}

void test_dm_get_col(void) {
  double values[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  DoubleMatrix *mat = dm_convert_array(3, 2, values);
  DoubleMatrix *vec = dm_get_col(mat, 1);
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(vec, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(vec, 1, 1));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(vec, 2, 1));
  TEST_ASSERT_EQUAL(3, vec->rows);
  dm_destroy(mat);
  dm_destroy(vec);
}

void test_dm_get_last_row(void) {
  double values[3][4] = {
      {1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.9, 10.0, 11.0, 12.0}};
  DoubleMatrix *mat = dm_convert_array(3, 4, values);
  DoubleMatrix *vec = dm_get_last_row(mat);
  TEST_ASSERT_EQUAL_DOUBLE(9.9, dm_get(vec, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(10.0, dm_get(vec, 0, 1));
  TEST_ASSERT_EQUAL_DOUBLE(11.0, dm_get(vec, 0, 2));
  TEST_ASSERT_EQUAL_DOUBLE(12.0, dm_get(vec, 0, 3));
  TEST_ASSERT_EQUAL(1, vec->rows);
  dm_destroy(mat);
  dm_destroy(vec);
}

void test_dm_get_last_col(void) {
  double values[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  DoubleMatrix *mat = dm_convert_array(3, 2, values);
  DoubleMatrix *vec = dm_get_last_col(mat);
  TEST_ASSERT_EQUAL_DOUBLE(2.0, dm_get(vec, 0, 0));
  TEST_ASSERT_EQUAL_DOUBLE(4.0, dm_get(vec, 1, 0));
  TEST_ASSERT_EQUAL_DOUBLE(6.0, dm_get(vec, 2, 0));
  TEST_ASSERT_EQUAL(3, vec->rows);
  dm_destroy(mat);
  dm_destroy(vec);
}

void test_dm_clone(void) {
  double values[3][2] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  DoubleMatrix *mat = dm_convert_array(3, 2, values);
  DoubleMatrix *clone = dm_clone(mat);
  TEST_ASSERT_TRUE(dm_equal(mat, clone));
  dm_destroy(mat);
  dm_destroy(clone);
}

void test_dm_identity(void) {
  size_t n = 3;
  DoubleMatrix *mat = dm_identity(n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      if (i == j) {
        TEST_ASSERT_EQUAL_DOUBLE(1.0, dm_get(mat, i, j));
      } else {
        TEST_ASSERT_EQUAL_DOUBLE(0.0, dm_get(mat, i, j));
      }
    }
  }
  dm_destroy(mat);
}

void test_dm_rand(void) {
  size_t rows = 3;
  size_t cols = 4;
  double density = 0.5;
  DoubleMatrix *mat = dm_rand(rows, cols, density);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      double value = dm_get(mat, i, j);
      TEST_ASSERT_TRUE(value >= 0.0 && value <= 1.0);
    }
  }
  dm_destroy(mat);
}

void test_dm_multiply(void) {
  double values1[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  double values2[3][2] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  double expected[2][2] = {{58.0, 64.0}, {139.0, 154.0}};
  DoubleMatrix *mat1 = dm_convert_array(2, 3, values1);
  DoubleMatrix *mat2 = dm_convert_array(3, 2, values2);
  DoubleMatrix *result = dm_multiply(mat1, mat2);
  DoubleMatrix *expected_mat = dm_convert_array(2, 2, expected);
  TEST_ASSERT_TRUE(dm_equal(result, expected_mat));
  dm_destroy(mat1);
  dm_destroy(mat2);
  dm_destroy(result);
  dm_destroy(expected_mat);
}

void test_dm_multiply_by_number(void) {
  double values[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  double expected[2][3] = {{2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}};
  DoubleMatrix *mat = dm_convert_array(2, 3, values);
  DoubleMatrix *result = dm_multiply_by_number(mat, 2.0);
  DoubleMatrix *expected_mat = dm_convert_array(2, 3, expected);
  TEST_ASSERT_TRUE(dm_equal(result, expected_mat));
  dm_destroy(mat);
  dm_destroy(result);
  dm_destroy(expected_mat);
}

void test_dm_transpose(void) {
  DoubleMatrix *mat = dm_create(2, 3);
  dm_set(mat, 0, 0, 1.0);
  dm_set(mat, 0, 1, 2.0);
  dm_set(mat, 0, 2, 3.0);
  dm_set(mat, 1, 0, 4.0);
  dm_set(mat, 1, 1, 5.0);
  dm_set(mat, 1, 2, 6.0);

  DoubleMatrix *transposed = dm_transpose(mat);

  TEST_ASSERT_NOT_NULL(transposed);
  TEST_ASSERT_EQUAL_INT(3, transposed->rows);
  TEST_ASSERT_EQUAL_INT(2, transposed->cols);

  double expected_data[] = {1, 4, 2, 5, 3, 6};
  for (int i = 0; i < 6; ++i) {
    TEST_ASSERT_EQUAL_DOUBLE(expected_data[i], transposed->values[i]);
  }

  dm_destroy(mat);
  dm_destroy(transposed);
}

void test_dm_add(void) {
  double values1[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  double values2[2][3] = {{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
  double expected[2][3] = {{8.0, 10.0, 12.0}, {14.0, 16.0, 18.0}};
  DoubleMatrix *mat1 = dm_convert_array(2, 3, values1);
  DoubleMatrix *mat2 = dm_convert_array(2, 3, values2);
  DoubleMatrix *result = dm_add(mat1, mat2);
  DoubleMatrix *expected_mat = dm_convert_array(2, 3, expected);
  TEST_ASSERT_TRUE(dm_equal(result, expected_mat));
  dm_destroy(mat1);
  dm_destroy(mat2);
  dm_destroy(result);
  dm_destroy(expected_mat);
}

void test_dm_diff(void) {
  double values1[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  double values2[2][3] = {{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
  double expected[2][3] = {{-6.0, -6.0, -6.0}, {-6.0, -6.0, -6.0}};
  DoubleMatrix *mat1 = dm_convert_array(2, 3, values1);
  DoubleMatrix *mat2 = dm_convert_array(2, 3, values2);
  DoubleMatrix *result = dm_diff(mat1, mat2);
  DoubleMatrix *expected_mat = dm_convert_array(2, 3, expected);
  TEST_ASSERT_TRUE(dm_equal(result, expected_mat));
  dm_destroy(mat1);
  dm_destroy(mat2);
  dm_destroy(result);
  dm_destroy(expected_mat);
}

void test_dm_inverse(void) {
  double values[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
  double expected[2][2] = {{-2.0, 1.0}, {1.5, -0.5}};
  DoubleMatrix *mat = dm_convert_array(2, 2, values);
  DoubleMatrix *result = dm_inverse(mat);
  DoubleMatrix *expected_mat = dm_convert_array(2, 2, expected);
  TEST_ASSERT_TRUE(dm_equal(result, expected_mat));
  dm_destroy(mat);
  dm_destroy(result);
  dm_destroy(expected_mat);
}

void test_dm_determinant(void) {
  double values[3][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  DoubleMatrix *mat = dm_convert_array(3, 3, values);
  double det = dm_determinant(mat);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, det);
  dm_destroy(mat);
}

void test_dm_trace(void) {
  double values[3][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  DoubleMatrix *mat = dm_convert_array(3, 3, values);
  double trace = dm_trace(mat);
  TEST_ASSERT_EQUAL_DOUBLE(15.0, trace);
  dm_destroy(mat);
}

void test_dm_rank(void) {
  double values[3][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  DoubleMatrix *mat = dm_convert_array(3, 3, values);
  size_t rank = dm_rank(mat);
  TEST_ASSERT_EQUAL(2, rank);
  dm_destroy(mat);
}

void test_dm_norm(void) {
  double values[2][2] = {{1.0, 2.0}, {3.0, 4.0}};
  DoubleMatrix *mat = dm_convert_array(2, 2, values);
  double norm = dm_norm(mat);
  TEST_ASSERT_EQUAL_DOUBLE(5.477225575051661, norm);
  dm_destroy(mat);
}

void test_dm_multiply_me_by_number(void) {
  double values[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  double expected[2][3] = {{2.0, 4.0, 6.0}, {8.0, 10.0, 12.0}};
  DoubleMatrix *mat = dm_convert_array(2, 3, values);
  dm_multiply_me_by_number(mat, 2.0);
  DoubleMatrix *expected_mat = dm_convert_array(2, 3, expected);
  TEST_ASSERT_TRUE(dm_equal(mat, expected_mat));
  dm_destroy(mat);
  dm_destroy(expected_mat);
}

void test_dm_equal(void) {
  double values1[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  double values2[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  DoubleMatrix *mat1 = dm_convert_array(2, 3, values1);
  DoubleMatrix *mat2 = dm_convert_array(2, 3, values2);
  TEST_ASSERT_TRUE(dm_equal(mat1, mat2));
  dm_destroy(mat1);
  dm_destroy(mat2);
}
