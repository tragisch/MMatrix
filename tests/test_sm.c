/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "m_io.h"
#include "sm.h"

/******************************
 ** Test preconditions:
 *******************************/
#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 5

/* Support for Meta Test Rig */
#define TEST_CASE(...)

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Creation of matrices:
 *******************************/

#define EPSILON 1e-5f

void setUp(void) {
  // Remove the test file if it exists
  // remove("test_matrix.mat");
}

// Teardown function called after each test
void tearDown(void) {
  // Remove the test file if it exists
  // remove("test_matrix.mat");
}

void test_sm_create(void) {

  // Test case 1: Create a matrix with valid dimensions
  size_t rows = 3;
  size_t cols = 4;
  FloatMatrix *matrix = sm_create(rows, cols);

  TEST_ASSERT_NOT_NULL_MESSAGE(matrix, "Failed to allocate matrix");
  TEST_ASSERT_EQUAL(rows, matrix->rows);
  TEST_ASSERT_EQUAL(cols, matrix->cols);
  TEST_ASSERT_EQUAL(rows * cols, matrix->capacity);

  for (size_t i = 0; i < matrix->rows; i++) {
    for (size_t j = 0; j < matrix->cols; j++) {
      TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0, sm_get(matrix, i, j));
    }
  }
  // Free the memory allocated for the matrix.
  sm_destroy(matrix);
}

void test_sm_create_from_array(void) {
  size_t rows = 2;
  size_t cols = 2;

  // Dynamisches 2D-Array erstellen
  float **array = (float **)malloc(rows * sizeof(float *));
  for (size_t i = 0; i < rows; ++i) {
    array[i] = (float *)malloc(cols * sizeof(float));
  }

  // Werte setzen
  array[0][0] = 1.0f;
  array[0][1] = 2.0f;
  array[1][0] = 3.0f;
  array[1][1] = 4.0f;

  // Matrix erstellen
  FloatMatrix *mat = sm_from_array_ptrs(rows, cols, array);
  TEST_ASSERT_NOT_NULL(mat);

  // Werte prüfen
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, sm_get(mat, 1, 1));

  // Aufräumen
  sm_destroy(mat);
  for (size_t i = 0; i < rows; ++i) {
    free(array[i]);
  }
  free(array);
}

void test_sm_create_from_2D_array(void) {
  float input[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};

  FloatMatrix *mat = sm_from_array_static(2, 2, input);
  TEST_ASSERT_NOT_NULL(mat);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, sm_get(mat, 1, 1));

  sm_destroy(mat);
}

void test_sm_convert_array(void) {
  float values[3][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  FloatMatrix *mat = sm_from_array_static(3, 2, values);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, sm_get(mat, 1, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, sm_get(mat, 2, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, sm_get(mat, 2, 1));
  sm_destroy(mat);
}

void test_sm_create_array_from_matrix(void) {
  float data[2][3] = {{1.5f, 2.5f, 3.5f}, {4.5f, 5.5f, 6.5f}};
  FloatMatrix *mat = sm_from_array_static(2, 3, data);
  TEST_ASSERT_NOT_NULL(mat);

  double *arr = sm_create_array_from_matrix(mat);
  TEST_ASSERT_NOT_NULL(arr);

  double expected[] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5};
  for (size_t i = 0; i < 6; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i], arr[i]);
  }

  free(arr);
  sm_destroy(mat);
}

void test_sm_set(void) {
  FloatMatrix *mat = sm_create(3, 3);
  sm_set(mat, 0, 0, 1.0f);
  sm_set(mat, 0, 1, 2.0f);
  sm_set(mat, 0, 2, 3.0f);
  sm_set(mat, 1, 0, 4.0f);
  sm_set(mat, 1, 1, 5.0f);
  sm_set(mat, 1, 2, 6.0f);
  sm_set(mat, 2, 0, 7.0f);
  sm_set(mat, 2, 1, 8.0f);
  sm_set(mat, 2, 2, 9.0f);

  // test if all values are set correctly
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, sm_get(mat, 0, 2));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, sm_get(mat, 1, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, sm_get(mat, 1, 2));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 7.0f, sm_get(mat, 2, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 8.0f, sm_get(mat, 2, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 9.0f, sm_get(mat, 2, 2));

  // Free the memory allocated for the matrix.
  sm_destroy(mat);
}

void test_sm_get(void) {
  float values[3][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  FloatMatrix *mat = sm_from_array_static(3, 2, values);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, sm_get(mat, 1, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, sm_get(mat, 2, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, sm_get(mat, 2, 1));
  sm_destroy(mat);
}

void test_sm_get_row() {
  // create test matrix
  float values[3][4] = {{1.0f, 2.0f, 3.0f, 4.0f},
                        {5.0f, 6.0f, 7.0f, 8.0f},
                        {9.9f, 10.0f, 11.0f, 12.0f}};
  FloatMatrix *mat = sm_from_array_static(3, 4, values);

  // get row vector
  FloatMatrix *vec = sm_get_row(mat, 1);

  // check vector length
  TEST_ASSERT_EQUAL_INT(1, vec->rows);

  // check vector values
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, sm_get(vec, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, sm_get(vec, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 7.0f, sm_get(vec, 0, 2));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 8.0f, sm_get(vec, 0, 3));

  // free memory
  sm_destroy(mat);
  sm_destroy(vec);
}

void test_sm_get_col(void) {
  float values[3][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  FloatMatrix *mat = sm_from_array_static(3, 2, values);
  FloatMatrix *vec = sm_get_col(mat, 1);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, sm_get(vec, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, sm_get(vec, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, sm_get(vec, 2, 0));
  TEST_ASSERT_EQUAL(3, vec->rows);
  sm_destroy(mat);
  sm_destroy(vec);
}

void test_sm_get_last_row(void) {
  float values[3][4] = {{1.0f, 2.0f, 3.0f, 4.0f},
                        {5.0f, 6.0f, 7.0f, 8.0f},
                        {9.9f, 10.0f, 11.0f, 12.0f}};
  FloatMatrix *mat = sm_from_array_static(3, 4, values);
  FloatMatrix *vec = sm_get_last_row(mat);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 9.9f, sm_get(vec, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 10.0f, sm_get(vec, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 11.0f, sm_get(vec, 0, 2));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 12.0f, sm_get(vec, 0, 3));
  TEST_ASSERT_EQUAL(1, vec->rows);
  sm_destroy(mat);
  sm_destroy(vec);
}

void test_sm_get_last_col(void) {
  float values[3][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  FloatMatrix *mat = sm_from_array_static(3, 2, values);
  FloatMatrix *vec = sm_get_last_col(mat);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, sm_get(vec, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, sm_get(vec, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, sm_get(vec, 2, 0));
  TEST_ASSERT_EQUAL(3, vec->rows);
  sm_destroy(mat);
  sm_destroy(vec);
}

void test_sm_clone(void) {
  float values[3][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  FloatMatrix *mat = sm_from_array_static(3, 2, values);
  FloatMatrix *clone = sm_clone(mat);
  TEST_ASSERT_TRUE(sm_is_equal(mat, clone));
  sm_destroy(mat);
  sm_destroy(clone);
}

void test_sm_identity(void) {
  size_t n = 3;
  FloatMatrix *mat = sm_create_identity(n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      if (i == j) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sm_get(mat, i, j));
      } else {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(mat, i, j));
      }
    }
  }
  sm_destroy(mat);
}

void test_sm_rand(void) {
  size_t rows = 3;
  size_t cols = 4;
  FloatMatrix *mat = sm_create_random(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      float value = sm_get(mat, i, j);
      TEST_ASSERT_TRUE(value >= 0.0f && value <= 1.0f);
    }
  }
  sm_destroy(mat);
}

void test_sm_random_xavier_distribution(void) {
  size_t rows = 100;
  size_t cols = 100;
  size_t fan_in = 128;
  size_t fan_out = 64;

  FloatMatrix *mat = sm_create_random_xavier(rows, cols, fan_in, fan_out);
  TEST_ASSERT_NOT_NULL(mat);

  size_t n = rows * cols;
  float sum = 0.0f;
  float sum_sq = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    float v = mat->values[i];
    sum += v;
    sum_sq += v * v;
  }

  float mean = sum / n;
  float variance = sum_sq / n - mean * mean;
  float stddev = sqrtf(variance);
  float expected_stddev = sqrtf(2.0f / (fan_in + fan_out));

  TEST_ASSERT_FLOAT_WITHIN(0.05f, 0.0f, mean); // Average_mean ≈ 0
  TEST_ASSERT_FLOAT_WITHIN(0.05f, expected_stddev, stddev);

  sm_destroy(mat);
}

void test_sm_random_he_distribution(void) {
  size_t rows = 100;
  size_t cols = 100;
  size_t fan_in = 128;

  FloatMatrix *mat = sm_create_random_he(rows, cols, fan_in);
  TEST_ASSERT_NOT_NULL(mat);

  size_t n = rows * cols;
  float sum = 0.0f;
  float sum_sq = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    float v = mat->values[i];
    sum += v;
    sum_sq += v * v;
  }

  float mean = sum / n;
  float variance = sum_sq / n - mean * mean;
  float stddev = sqrtf(variance);
  float expected_stddev = sqrtf(2.0f / fan_in);

  // Toleranzen: etwas großzügig, da Zufall im Spiel ist
  TEST_ASSERT_FLOAT_WITHIN(0.05f, 0.0f, mean); // Mittelwert ≈ 0
  TEST_ASSERT_FLOAT_WITHIN(0.05f, expected_stddev,
                           stddev); // Varianz wie erwartet

  sm_destroy(mat);
}

void test_sm_multiply(void) {
  float values1[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float values2[3][2] = {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}};
  float expected[2][2] = {{58.0f, 64.0f}, {139.0f, 154.0f}};
  FloatMatrix *mat1 = sm_from_array_static(2, 3, values1);
  FloatMatrix *mat2 = sm_from_array_static(3, 2, values2);
  FloatMatrix *result = sm_multiply(mat1, mat2);
  FloatMatrix *expected_mat = sm_from_array_static(2, 2, expected);
  TEST_ASSERT_TRUE(sm_is_equal(result, expected_mat));
  sm_destroy(mat1);
  sm_destroy(mat2);
  sm_destroy(result);
  sm_destroy(expected_mat);
}

void test_sm_multiply_by_number(void) {
  float values[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float expected[2][3] = {{2.0f, 4.0f, 6.0f}, {8.0f, 10.0f, 12.0f}};
  FloatMatrix *mat = sm_from_array_static(2, 3, values);
  FloatMatrix *result = sm_multiply_by_number(mat, 2.0f);
  FloatMatrix *expected_mat = sm_from_array_static(2, 3, expected);
  TEST_ASSERT_TRUE(sm_is_equal(result, expected_mat));
  sm_destroy(mat);
  sm_destroy(result);
  sm_destroy(expected_mat);
}

void test_sm_transpose(void) {
  FloatMatrix *mat = sm_create(2, 3);
  sm_set(mat, 0, 0, 1.0f);
  sm_set(mat, 0, 1, 2.0f);
  sm_set(mat, 0, 2, 3.0f);
  sm_set(mat, 1, 0, 4.0f);
  sm_set(mat, 1, 1, 5.0f);
  sm_set(mat, 1, 2, 6.0f);

  FloatMatrix *transposed = sm_transpose(mat);

  TEST_ASSERT_NOT_NULL(transposed);
  TEST_ASSERT_EQUAL_INT(3, transposed->rows);
  TEST_ASSERT_EQUAL_INT(2, transposed->cols);

  float expected_data[] = {1, 4, 2, 5, 3, 6};
  for (int i = 0; i < 6; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected_data[i], transposed->values[i]);
  }

  sm_destroy(mat);
  sm_destroy(transposed);
}

void test_sm_add(void) {
  float values1[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float values2[2][3] = {{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}};
  float expected[2][3] = {{8.0f, 10.0f, 12.0f}, {14.0f, 16.0f, 18.0f}};
  FloatMatrix *mat1 = sm_from_array_static(2, 3, values1);
  FloatMatrix *mat2 = sm_from_array_static(2, 3, values2);
  FloatMatrix *result = sm_add(mat1, mat2);
  FloatMatrix *expected_mat = sm_from_array_static(2, 3, expected);
  TEST_ASSERT_TRUE(sm_is_equal(result, expected_mat));
  sm_destroy(mat1);
  sm_destroy(mat2);
  sm_destroy(result);
  sm_destroy(expected_mat);
}

void test_sm_diff(void) {
  float values1[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float values2[2][3] = {{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}};
  float expected[2][3] = {{-6.0f, -6.0f, -6.0f}, {-6.0f, -6.0f, -6.0f}};
  FloatMatrix *mat1 = sm_from_array_static(2, 3, values1);
  FloatMatrix *mat2 = sm_from_array_static(2, 3, values2);
  FloatMatrix *result = sm_diff(mat1, mat2);
  FloatMatrix *expected_mat = sm_from_array_static(2, 3, expected);
  TEST_ASSERT_TRUE(sm_is_equal(result, expected_mat));
  sm_destroy(mat1);
  sm_destroy(mat2);
  sm_destroy(result);
  sm_destroy(expected_mat);
}

void test_sm_trace(void) {
  float values[3][3] = {
      {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}};
  FloatMatrix *mat = sm_from_array_static(3, 3, values);
  float trace = sm_trace(mat);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 15.0f, trace);
  sm_destroy(mat);
}

void test_sm_norm(void) {
  float values[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  FloatMatrix *mat = sm_from_array_static(2, 2, values);
  float norm = sm_norm(mat);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.4772256f, norm);
  sm_destroy(mat);
}

void test_sm_equal(void) {
  float values1[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float values2[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  FloatMatrix *mat1 = sm_from_array_static(2, 3, values1);
  FloatMatrix *mat2 = sm_from_array_static(2, 3, values2);
  TEST_ASSERT_TRUE(sm_is_equal(mat1, mat2));
  sm_destroy(mat1);
  sm_destroy(mat2);
}

// Helper function to create a sample matrix
FloatMatrix *create_sample_matrix(size_t rows, size_t cols) {
  FloatMatrix *matrix = sm_create(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      sm_set(matrix, i, j, (float)(i * cols + j));
    }
  }
  return matrix;
}

void sm_back_substitution(const FloatMatrix *mat, float *solution) {
  size_t rows = mat->rows;
  size_t cols = mat->cols;

  // Rückwärtseinsetzen
  for (int i = rows - 1; i >= 0; i--) {
    float sum = 0.0f;
    for (size_t j = i + 1; j < cols - 1; j++) {
      sum += sm_get(mat, i, j) * solution[j];
    }
    solution[i] = (sm_get(mat, i, cols - 1) - sum) / sm_get(mat, i, i);
  }
}

// void test_sm_linear_batch_should_multiply_and_add_bias(void) {
//   float input[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
//   float weights[2][3] = {{1.0f, 0.0f, -1.0f}, {0.5f, 0.5f, 0.5f}};
//   float bias[1][2] = {{1.0f, 2.0f}};
//   float expected[2][2] = {
//       {1.0f + (1 * 1 + 2 * 0 + 3 * -1), 2.0f + (1 * 0.5 + 2 * 0.5 + 3 *
//       0.5)}, {1.0f + (4 * 1 + 5 * 0 + 6 * -1), 2.0f + (4 * 0.5 + 5 * 0.5 + 6
//       * 0.5)}};

//   FloatMatrix *inputs = sm_create_from_2D_array(2, 3, input);
//   FloatMatrix *W = sm_create_from_2D_array(2, 3, weights);
//   FloatMatrix *b = sm_create_from_2D_array(1, 2, bias);

//   FloatMatrix *output = sm_linear_batch(inputs, W, b);
//   FloatMatrix *expected_mat = sm_create_from_2D_array(2, 2, expected);

//   TEST_ASSERT_TRUE(sm_is_equal(output, expected_mat));

//   sm_destroy(inputs);
//   sm_destroy(W);
//   sm_destroy(b);
//   sm_destroy(output);
//   sm_destroy(expected_mat);
// }

// // Helper function to create a matrix from a flat array
// FloatMatrix *sm_create_from_flat_array(size_t rows, size_t cols,
//                                        const float *flat) {
//   FloatMatrix *mat = sm_create(rows, cols);
//   if (!mat)
//     return NULL;
//   memcpy(mat->values, flat, sizeof(float) * rows * cols);
//   return mat;
// }

//