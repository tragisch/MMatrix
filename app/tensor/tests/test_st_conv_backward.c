/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_conv.h"

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

void test_st_conv2d_backward_data_nchw_identity_weight(void) {
  FloatTensor *grad_output = create_4d(1, 1, 3, 3);
  FloatTensor *weight = create_4d(1, 1, 1, 1);
  FloatTensor *grad_input = create_4d(1, 1, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(grad_input);

  for (size_t i = 0; i < 9; ++i) {
    grad_output->values[i] = (float)(i + 1);
  }
  weight->values[0] = 1.0f;

  StConv2dParams p = st_conv2d_default_params();
  bool ok = st_conv2d_backward_data_nchw(grad_output, weight, &p, grad_input);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 9; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, (float)(i + 1), grad_input->values[i]);
  }

  st_destroy(grad_input);
  st_destroy(weight);
  st_destroy(grad_output);
}

void test_st_conv2d_backward_data_nchw_3x3_kernel(void) {
  FloatTensor *grad_output = create_4d(1, 1, 2, 2);
  FloatTensor *weight = create_4d(1, 1, 2, 2);
  FloatTensor *grad_input = create_4d(1, 1, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(grad_input);

  float go[] = {1, 2, 3, 4};
  memcpy(grad_output->values, go, sizeof(go));

  float wt[] = {1, 0, 0, 1};
  memcpy(weight->values, wt, sizeof(wt));

  StConv2dParams p = st_conv2d_default_params();
  bool ok = st_conv2d_backward_data_nchw(grad_output, weight, &p, grad_input);
  TEST_ASSERT_TRUE(ok);

  float expected[] = {1, 2, 0, 3, 5, 2, 0, 3, 4};
  for (size_t i = 0; i < 9; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i], grad_input->values[i]);
  }

  st_destroy(grad_input);
  st_destroy(weight);
  st_destroy(grad_output);
}

void test_st_conv2d_backward_weight_nchw_basic(void) {
  FloatTensor *input = create_4d(1, 1, 3, 3);
  FloatTensor *grad_output = create_4d(1, 1, 2, 2);
  FloatTensor *grad_weight = create_4d(1, 1, 2, 2);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(grad_weight);

  for (size_t i = 0; i < 9; ++i) {
    input->values[i] = (float)(i + 1);
  }

  for (size_t i = 0; i < 4; ++i) {
    grad_output->values[i] = 1.0f;
  }

  StConv2dParams p = st_conv2d_default_params();
  bool ok =
      st_conv2d_backward_weight_nchw(input, grad_output, &p, grad_weight);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 12.0f, grad_weight->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 16.0f, grad_weight->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 24.0f, grad_weight->values[2]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 28.0f, grad_weight->values[3]);

  st_destroy(grad_weight);
  st_destroy(grad_output);
  st_destroy(input);
}

void test_st_conv2d_backward_bias_should_sum_over_n_h_w(void) {
  FloatTensor *grad_output = create_4d(2, 3, 2, 2);
  FloatTensor *grad_bias = create_1d(3);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(grad_bias);

  for (size_t ni = 0; ni < 2; ++ni) {
    for (size_t ci = 0; ci < 3; ++ci) {
      for (size_t i = 0; i < 4; ++i) {
        grad_output->values[((ni * 3 + ci) * 2 * 2) + i] =
            (float)(ci + 1);
      }
    }
  }

  bool ok = st_conv2d_backward_bias(grad_output, grad_bias);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 8.0f, grad_bias->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 16.0f, grad_bias->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 24.0f, grad_bias->values[2]);

  st_destroy(grad_bias);
  st_destroy(grad_output);
}

void test_st_conv2d_backward_gradient_consistency(void) {
  FloatTensor *input = create_4d(1, 1, 3, 3);
  FloatTensor *weight = create_4d(1, 1, 2, 2);
  FloatTensor *output = create_4d(1, 1, 2, 2);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  for (size_t i = 0; i < 9; ++i) {
    input->values[i] = (float)(i + 1) * 0.1f;
  }
  float w_init[] = {0.5f, -0.3f, 0.2f, 0.8f};
  memcpy(weight->values, w_init, sizeof(w_init));

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_REFERENCE;

  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  FloatTensor *grad_output = create_4d(1, 1, 2, 2);
  FloatTensor *grad_weight = create_4d(1, 1, 2, 2);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(grad_weight);

  for (size_t i = 0; i < 4; ++i) {
    grad_output->values[i] = 1.0f;
  }

  ok = st_conv2d_backward_weight_nchw(input, grad_output, &p, grad_weight);
  TEST_ASSERT_TRUE(ok);

  const float eps = 1e-3f;
  for (size_t wi = 0; wi < 4; ++wi) {
    float orig = weight->values[wi];

    weight->values[wi] = orig + eps;
    ok = st_conv2d_nchw(input, weight, NULL, &p, output);
    TEST_ASSERT_TRUE(ok);
    float loss_plus = 0;
    for (size_t i = 0; i < 4; ++i) loss_plus += output->values[i];

    weight->values[wi] = orig - eps;
    ok = st_conv2d_nchw(input, weight, NULL, &p, output);
    TEST_ASSERT_TRUE(ok);
    float loss_minus = 0;
    for (size_t i = 0; i < 4; ++i) loss_minus += output->values[i];

    weight->values[wi] = orig;

    float numerical_grad = (loss_plus - loss_minus) / (2.0f * eps);
    TEST_ASSERT_FLOAT_WITHIN(1e-2f, numerical_grad, grad_weight->values[wi]);
  }

  st_destroy(grad_weight);
  st_destroy(grad_output);
  st_destroy(output);
  st_destroy(weight);
  st_destroy(input);
}
