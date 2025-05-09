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
  FloatMatrix *mat = sm_create_from_2D_array(2, 2, values);
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
  FloatMatrix *mat = sm_create_from_2D_array(2, 2, values);
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
  FloatMatrix *mat = sm_create_from_2D_array(2, 2, values);
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
  FloatMatrix *mat = sm_create_from_2D_array(1, 4, values);
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
