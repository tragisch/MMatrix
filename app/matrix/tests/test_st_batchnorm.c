/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_batchnorm.h"

#include <math.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 6

/* Support for Meta Test Rig */
#define TEST_CASE(...)

#if __has_include("unity.h")
#include "unity.h"
#endif

#ifndef TEST_ASSERT_NOT_NULL
#define TEST_ASSERT_NOT_NULL(value) \
  do {                             \
    (void)(value);                 \
  } while (0)
#endif
#ifndef TEST_ASSERT_EQUAL
#define TEST_ASSERT_EQUAL(expected, actual) \
  do {                                      \
    (void)(expected);                       \
    (void)(actual);                         \
  } while (0)
#endif
#ifndef TEST_ASSERT_EQUAL_UINT32
#define TEST_ASSERT_EQUAL_UINT32(expected, actual) \
  do {                                             \
    (void)(expected);                              \
    (void)(actual);                                \
  } while (0)
#endif
#ifndef TEST_ASSERT_TRUE
#define TEST_ASSERT_TRUE(condition) \
  do {                              \
    (void)(condition);              \
  } while (0)
#endif
#ifndef TEST_ASSERT_FALSE
#define TEST_ASSERT_FALSE(condition) \
  do {                               \
    (void)(condition);               \
  } while (0)
#endif
#ifndef TEST_ASSERT_FLOAT_WITHIN
#define TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual) \
  do {                                                    \
    (void)(delta);                                        \
    (void)(expected);                                     \
    (void)(actual);                                       \
  } while (0)
#endif

#define EPSILON 1e-4f

void setUp(void) {}
void tearDown(void) {}

static FloatTensor *create_4d(size_t n, size_t c, size_t h, size_t w) {
  size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

static FloatTensor *create_1d(size_t len) {
  size_t shape[1] = {len};
  return st_create(1, shape);
}

/* ---- Forward ---- */

void test_st_batchnorm2d_forward_uniform_input_should_output_zero(void) {
  /* If all inputs for a channel are the same, output should be beta. */
  FloatTensor *input = create_4d(2, 1, 2, 2);
  FloatTensor *output = create_4d(2, 1, 2, 2);
  FloatTensor *mean = create_1d(1);
  FloatTensor *var = create_1d(1);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);

  /* All elements = 5.0 */
  for (size_t i = 0; i < 8; ++i) {
    input->values[i] = 5.0f;
  }

  bool ok = st_batchnorm2d_forward(input, NULL, NULL, 1e-5f, output, mean, var);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, mean->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, var->values[0]);

  /* Output should be (5-5)/sqrt(0+eps)*1 + 0 ≈ 0 */
  for (size_t i = 0; i < 8; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-2f, 0.0f, output->values[i]);
  }

  st_destroy(var);
  st_destroy(mean);
  st_destroy(output);
  st_destroy(input);
}

void test_st_batchnorm2d_forward_with_gamma_beta(void) {
  FloatTensor *input = create_4d(1, 2, 1, 1);
  FloatTensor *output = create_4d(1, 2, 1, 1);
  FloatTensor *mean = create_1d(2);
  FloatTensor *var = create_1d(2);
  FloatTensor *gamma = create_1d(2);
  FloatTensor *beta = create_1d(2);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);
  TEST_ASSERT_NOT_NULL(gamma);
  TEST_ASSERT_NOT_NULL(beta);

  input->values[0] = 1.0f;
  input->values[1] = 2.0f;
  gamma->values[0] = 2.0f;
  gamma->values[1] = 3.0f;
  beta->values[0] = 10.0f;
  beta->values[1] = 20.0f;

  bool ok = st_batchnorm2d_forward(input, gamma, beta, 1e-5f, output, mean,
                                   var);
  TEST_ASSERT_TRUE(ok);

  /* With single sample per channel, x_hat = 0, output = beta. */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 10.0f, output->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 20.0f, output->values[1]);

  st_destroy(beta);
  st_destroy(gamma);
  st_destroy(var);
  st_destroy(mean);
  st_destroy(output);
  st_destroy(input);
}

void test_st_batchnorm2d_forward_known_values(void) {
  /* 2 batches, 1 channel, 1x2 spatial: values [1,3] and [5,7]
   * mean = (1+3+5+7)/4 = 4
   * var  = ((1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2)/4 = (9+1+1+9)/4 = 5
   * x_hat = (x - 4) / sqrt(5 + 1e-5)
   */
  FloatTensor *input = create_4d(2, 1, 1, 2);
  FloatTensor *output = create_4d(2, 1, 1, 2);
  FloatTensor *mean = create_1d(1);
  FloatTensor *var = create_1d(1);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);

  input->values[0] = 1.0f;
  input->values[1] = 3.0f;
  input->values[2] = 5.0f;
  input->values[3] = 7.0f;

  bool ok = st_batchnorm2d_forward(input, NULL, NULL, 1e-5f, output, mean, var);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, mean->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, var->values[0]);

  float inv_std = 1.0f / sqrtf(5.0f + 1e-5f);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, (1.0f - 4.0f) * inv_std,
                           output->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, (3.0f - 4.0f) * inv_std,
                           output->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, (5.0f - 4.0f) * inv_std,
                           output->values[2]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, (7.0f - 4.0f) * inv_std,
                           output->values[3]);

  st_destroy(var);
  st_destroy(mean);
  st_destroy(output);
  st_destroy(input);
}

/* ---- Backward ---- */

void test_st_batchnorm2d_backward_gradient_consistency(void) {
  /* Numerical gradient check for batchnorm backward. */
  FloatTensor *input = create_4d(2, 1, 1, 2);
  FloatTensor *output = create_4d(2, 1, 1, 2);
  FloatTensor *mean = create_1d(1);
  FloatTensor *var = create_1d(1);
  FloatTensor *grad_output = create_4d(2, 1, 1, 2);
  FloatTensor *grad_input = create_4d(2, 1, 1, 2);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(grad_input);

  input->values[0] = 1.0f;
  input->values[1] = 3.0f;
  input->values[2] = 5.0f;
  input->values[3] = 7.0f;

  float eps = 1e-5f;

  /* Forward */
  bool ok = st_batchnorm2d_forward(input, NULL, NULL, eps, output, mean, var);
  TEST_ASSERT_TRUE(ok);

  /* grad_output = all 1.0 (L = sum(output)) */
  for (size_t i = 0; i < 4; ++i) {
    grad_output->values[i] = 1.0f;
  }

  /* Backward */
  ok = st_batchnorm2d_backward(grad_output, input, mean, var, NULL, eps,
                               grad_input, NULL, NULL);
  TEST_ASSERT_TRUE(ok);

  /* Numerical gradient check */
  float h_num = 1e-3f;
  for (size_t idx = 0; idx < 4; ++idx) {
    float orig = input->values[idx];

    input->values[idx] = orig + h_num;
    ok = st_batchnorm2d_forward(input, NULL, NULL, eps, output, mean, var);
    TEST_ASSERT_TRUE(ok);
    float loss_plus = 0;
    for (size_t i = 0; i < 4; ++i) loss_plus += output->values[i];

    input->values[idx] = orig - h_num;
    ok = st_batchnorm2d_forward(input, NULL, NULL, eps, output, mean, var);
    TEST_ASSERT_TRUE(ok);
    float loss_minus = 0;
    for (size_t i = 0; i < 4; ++i) loss_minus += output->values[i];

    input->values[idx] = orig;

    float numerical = (loss_plus - loss_minus) / (2.0f * h_num);
    TEST_ASSERT_FLOAT_WITHIN(5e-2f, numerical, grad_input->values[idx]);
  }

  st_destroy(grad_input);
  st_destroy(grad_output);
  st_destroy(var);
  st_destroy(mean);
  st_destroy(output);
  st_destroy(input);
}


