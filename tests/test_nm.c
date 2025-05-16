
/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "nm.h"
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

void test_nm_apply_relu_should_set_negatives_to_zero(void) {
  float values[2][2] = {{-2.0f, 0.0f}, {3.0f, -1.5f}};
  FloatMatrix *mat = sm_from_array_static(2, 2, values);
  TEST_ASSERT_NOT_NULL(mat);

  nm_apply_relu(mat);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(mat, 1, 1));

  sm_destroy(mat);
}

void test_nm_apply_sigmoid_should_map_to_range_0_to_1(void) {
  float values[2][2] = {{-2.0f, 0.0f}, {2.0f, 4.0f}};
  FloatMatrix *mat = sm_from_array_static(2, 2, values);
  TEST_ASSERT_NOT_NULL(mat);

  nm_apply_sigmoid(mat);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f / (1.0f + expf(2.0f)),
                           sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.5f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f / (1.0f + expf(-2.0f)),
                           sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f / (1.0f + expf(-4.0f)),
                           sm_get(mat, 1, 1));

  sm_destroy(mat);
}

void test_nm_apply_tanh_should_map_to_range_minus1_to_1(void) {
  float values[2][2] = {{-2.0f, 0.0f}, {2.0f, 4.0f}};
  FloatMatrix *mat = sm_from_array_static(2, 2, values);
  TEST_ASSERT_NOT_NULL(mat);

  nm_apply_tanh(mat);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, tanhf(-2.0f), sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, tanhf(2.0f), sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, tanhf(4.0f), sm_get(mat, 1, 1));

  sm_destroy(mat);
}

void test_nm_apply_softmax_should_yield_probabilities_that_sum_to_1(void) {
  float values[1][4] = {{1.0f, 2.0f, 3.0f, 4.0f}};
  FloatMatrix *mat = sm_from_array_static(1, 4, values);
  TEST_ASSERT_NOT_NULL(mat);

  nm_apply_softmax(mat);

  float sum = 0.0f;
  for (size_t i = 0; i < 4; ++i) {
    float val = sm_get(mat, 0, i);
    TEST_ASSERT_TRUE(val >= 0.0f && val <= 1.0f);
    sum += val;
  }
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, sum);

  sm_destroy(mat);
}

void test_nm_inplace_add_rowwise_should_add_row_to_all_rows(void) {
  float values[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float bias[1][3] = {{10.0f, 20.0f, 30.0f}};

  FloatMatrix *mat = sm_from_array_static(2, 3, values);
  FloatMatrix *bias_row = sm_from_array_static(1, 3, bias);

  nm_inplace_add_rowwise(mat, bias_row);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 11.0f, sm_get(mat, 0, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 22.0f, sm_get(mat, 0, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 33.0f, sm_get(mat, 0, 2));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 14.0f, sm_get(mat, 1, 0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 25.0f, sm_get(mat, 1, 1));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 36.0f, sm_get(mat, 1, 2));

  sm_destroy(mat);
  sm_destroy(bias_row);
}

void test_nm_linear_should_perform_multiplication_and_add_bias(void) {
  float input_vals[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  float weight_vals[2][2] = {{5.0f, 6.0f}, {7.0f, 8.0f}};
  float bias_vals[1][2] = {{9.0f, 10.0f}};

  FloatMatrix *input = sm_from_array_static(2, 2, input_vals);
  FloatMatrix *weights = sm_from_array_static(2, 2, weight_vals);
  FloatMatrix *bias = sm_from_array_static(1, 2, bias_vals);

  FloatMatrix *result = nm_linear(input, weights, bias);
  TEST_ASSERT_NOT_NULL(result);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5 * 1 + 7 * 2 + 9,
                           sm_get(result, 0, 0)); // 5+14+9 = 28
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6 * 1 + 8 * 2 + 10,
                           sm_get(result, 0, 1)); // 6+16+10 = 32
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5 * 3 + 7 * 4 + 9,
                           sm_get(result, 1, 0)); // 15+28+9 = 52
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6 * 3 + 8 * 4 + 10,
                           sm_get(result, 1, 1)); // 18+32+10 = 60

  sm_destroy(input);
  sm_destroy(weights);
  sm_destroy(bias);
  sm_destroy(result);
}

void test_nm_mse_loss_should_return_mean_squared_error(void) {
  float pred_vals[2][2] = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  float target_vals[2][2] = {{1.0f, 2.0f}, {2.0f, 4.0f}};
  FloatMatrix *pred = sm_from_array_static(2, 2, pred_vals);
  FloatMatrix *target = sm_from_array_static(2, 2, target_vals);

  float mse = nm_mse_loss(pred, target);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.25f, mse); // (0^2 + 0^2 + 1^2 + 0^2)/4
}

void test_nm_cross_entropy_loss_should_return_correct_value(void) {
  float pred_vals[1][3] = {{0.7f, 0.2f, 0.1f}};
  float target_vals[1][3] = {{1.0f, 0.0f, 0.0f}};
  FloatMatrix *pred = sm_from_array_static(1, 3, pred_vals);
  FloatMatrix *target = sm_from_array_static(1, 3, target_vals);

  float ce = nm_cross_entropy_loss(pred, target);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, -logf(0.7f), ce);
}

void test_nm_softmax_denominator_should_compute_correct_sum(void) {
  float vec[3] = {1.0f, 2.0f, 3.0f};
  float expected = expf(1.0f - 3.0f) + expf(2.0f - 3.0f) + expf(3.0f - 3.0f);
  float result = nm_softmax_denominator(vec, 3);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected, result);
}

void test_nm_active_library_should_return_non_null(void) {
  const char *lib = nm_active_library();
  printf("Active library: %s\n", lib);
  TEST_ASSERT_NOT_NULL(lib);
}
//