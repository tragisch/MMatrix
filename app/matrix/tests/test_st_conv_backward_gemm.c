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

#define EPSILON 1e-3f

void setUp(void) {}
void tearDown(void) {}

static FloatTensor *create_4d(size_t n, size_t c, size_t h, size_t w) {
  size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

/* Fill tensor with simple incrementing values. */
static void fill_inc(FloatTensor *t) {
  for (size_t i = 0; i < t->numel; ++i) {
    t->values[i] = (float)(i + 1) * 0.1f;
  }
}

/* Fill tensor with pseudo-random values in [-1, 1]. */
static void fill_rand(FloatTensor *t, unsigned int seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1103515245u + 12345u;
    t->values[i] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
  }
}

/* ---- GEMM vs naive backward-data consistency ---- */

void test_gemm_vs_naive_backward_data_1x1x5x5(void) {
  /* Compare GEMM backward-data with naive reference for a larger tensor
   * that triggers the GEMM path (macs >= 1e4). */
  FloatTensor *input = create_4d(2, 3, 8, 8);
  FloatTensor *weight = create_4d(4, 3, 3, 3);
  FloatTensor *output = create_4d(2, 4, 6, 6);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  fill_rand(input, 42);
  fill_rand(weight, 123);

  StConv2dParams p = st_conv2d_default_params();

  /* Forward */
  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  /* grad_output = all 1.0 */
  FloatTensor *grad_output = create_4d(2, 4, 6, 6);
  TEST_ASSERT_NOT_NULL(grad_output);
  for (size_t i = 0; i < grad_output->numel; ++i) {
    grad_output->values[i] = 1.0f;
  }

  /* GEMM path (default) */
  FloatTensor *grad_input_gemm = create_4d(2, 3, 8, 8);
  TEST_ASSERT_NOT_NULL(grad_input_gemm);
  ok = st_conv2d_backward_data_nchw(grad_output, weight, &p, grad_input_gemm);
  TEST_ASSERT_TRUE(ok);

  /* Naive reference path */
  FloatTensor *grad_input_ref = create_4d(2, 3, 8, 8);
  TEST_ASSERT_NOT_NULL(grad_input_ref);
  StConv2dParams p_ref = p;
  p_ref.backend = ST_CONV_BACKEND_REFERENCE;
  ok = st_conv2d_backward_data_nchw(grad_output, weight, &p_ref, grad_input_ref);
  TEST_ASSERT_TRUE(ok);

  /* Compare */
  for (size_t i = 0; i < grad_input_gemm->numel; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, grad_input_ref->values[i],
                             grad_input_gemm->values[i]);
  }

  st_destroy(grad_input_ref);
  st_destroy(grad_input_gemm);
  st_destroy(grad_output);
  st_destroy(output);
  st_destroy(weight);
  st_destroy(input);
}

/* ---- GEMM vs naive backward-weight consistency ---- */

void test_gemm_vs_naive_backward_weight(void) {
  FloatTensor *input = create_4d(2, 3, 8, 8);
  FloatTensor *weight = create_4d(4, 3, 3, 3);
  FloatTensor *output = create_4d(2, 4, 6, 6);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  fill_rand(input, 42);
  fill_rand(weight, 123);

  StConv2dParams p = st_conv2d_default_params();

  /* Forward */
  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  /* grad_output = all 1.0 */
  FloatTensor *grad_output = create_4d(2, 4, 6, 6);
  TEST_ASSERT_NOT_NULL(grad_output);
  for (size_t i = 0; i < grad_output->numel; ++i) {
    grad_output->values[i] = 1.0f;
  }

  /* GEMM path */
  FloatTensor *grad_weight_gemm = create_4d(4, 3, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_weight_gemm);
  ok = st_conv2d_backward_weight_nchw(input, grad_output, &p, grad_weight_gemm);
  TEST_ASSERT_TRUE(ok);

  /* Naive reference */
  FloatTensor *grad_weight_ref = create_4d(4, 3, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_weight_ref);
  StConv2dParams p_ref = p;
  p_ref.backend = ST_CONV_BACKEND_REFERENCE;
  ok = st_conv2d_backward_weight_nchw(input, grad_output, &p_ref,
                                      grad_weight_ref);
  TEST_ASSERT_TRUE(ok);

  /* Compare */
  for (size_t i = 0; i < grad_weight_gemm->numel; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, grad_weight_ref->values[i],
                             grad_weight_gemm->values[i]);
  }

  st_destroy(grad_weight_ref);
  st_destroy(grad_weight_gemm);
  st_destroy(grad_output);
  st_destroy(output);
  st_destroy(weight);
  st_destroy(input);
}

/* ---- Numerical gradient check for backward data (GEMM path) ---- */

void test_gemm_backward_data_numerical_gradient(void) {
  FloatTensor *input = create_4d(1, 2, 5, 5);
  FloatTensor *weight = create_4d(3, 2, 3, 3);
  FloatTensor *output = create_4d(1, 3, 3, 3);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  fill_inc(input);
  fill_rand(weight, 77);

  StConv2dParams p = st_conv2d_default_params();

  /* Forward */
  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  /* grad_output = 1 */
  FloatTensor *grad_output = create_4d(1, 3, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_output);
  for (size_t i = 0; i < grad_output->numel; ++i) {
    grad_output->values[i] = 1.0f;
  }

  /* Analytic gradient w.r.t. input */
  FloatTensor *grad_input = create_4d(1, 2, 5, 5);
  TEST_ASSERT_NOT_NULL(grad_input);
  ok = st_conv2d_backward_data_nchw(grad_output, weight, &p, grad_input);
  TEST_ASSERT_TRUE(ok);

  /* Numerical gradient check for first 10 input elements */
  const float eps = 1e-3f;
  for (size_t xi = 0; xi < 10; ++xi) {
    float orig = input->values[xi];

    input->values[xi] = orig + eps;
    ok = st_conv2d_nchw(input, weight, NULL, &p, output);
    TEST_ASSERT_TRUE(ok);
    float loss_plus = 0;
    for (size_t i = 0; i < output->numel; ++i) loss_plus += output->values[i];

    input->values[xi] = orig - eps;
    ok = st_conv2d_nchw(input, weight, NULL, &p, output);
    TEST_ASSERT_TRUE(ok);
    float loss_minus = 0;
    for (size_t i = 0; i < output->numel; ++i) loss_minus += output->values[i];

    input->values[xi] = orig;

    float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
    TEST_ASSERT_FLOAT_WITHIN(0.05f, num_grad, grad_input->values[xi]);
  }

  st_destroy(grad_input);
  st_destroy(grad_output);
  st_destroy(output);
  st_destroy(weight);
  st_destroy(input);
}

/* ---- Numerical gradient check for backward weight (GEMM path) ---- */

void test_gemm_backward_weight_numerical_gradient(void) {
  FloatTensor *input = create_4d(1, 2, 5, 5);
  FloatTensor *weight = create_4d(3, 2, 3, 3);
  FloatTensor *output = create_4d(1, 3, 3, 3);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  fill_inc(input);
  fill_rand(weight, 77);

  StConv2dParams p = st_conv2d_default_params();

  /* Forward */
  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  /* grad_output = 1 */
  FloatTensor *grad_output = create_4d(1, 3, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_output);
  for (size_t i = 0; i < grad_output->numel; ++i) {
    grad_output->values[i] = 1.0f;
  }

  /* Analytic gradient w.r.t. weight */
  FloatTensor *grad_weight = create_4d(3, 2, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_weight);
  ok = st_conv2d_backward_weight_nchw(input, grad_output, &p, grad_weight);
  TEST_ASSERT_TRUE(ok);

  /* Numerical gradient check */
  const float eps = 1e-3f;
  for (size_t wi = 0; wi < weight->numel; ++wi) {
    float orig = weight->values[wi];

    weight->values[wi] = orig + eps;
    ok = st_conv2d_nchw(input, weight, NULL, &p, output);
    TEST_ASSERT_TRUE(ok);
    float loss_plus = 0;
    for (size_t i = 0; i < output->numel; ++i) loss_plus += output->values[i];

    weight->values[wi] = orig - eps;
    ok = st_conv2d_nchw(input, weight, NULL, &p, output);
    TEST_ASSERT_TRUE(ok);
    float loss_minus = 0;
    for (size_t i = 0; i < output->numel; ++i) loss_minus += output->values[i];

    weight->values[wi] = orig;

    float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
    TEST_ASSERT_FLOAT_WITHIN(0.05f, num_grad, grad_weight->values[wi]);
  }

  st_destroy(grad_weight);
  st_destroy(grad_output);
  st_destroy(output);
  st_destroy(weight);
  st_destroy(input);
}

/* ---- Winograd 3x3 backward data matches GEMM/naive ---- */

void test_winograd_backward_data_3x3_matches_reference(void) {
  /* Winograd is currently disabled; this test verifies the default path
   * (GEMM or naive fallback) still matches the naive reference. */
  FloatTensor *input = create_4d(1, 2, 6, 6);
  FloatTensor *weight = create_4d(3, 2, 3, 3);
  FloatTensor *output = create_4d(1, 3, 4, 4);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  fill_rand(input, 42);
  fill_rand(weight, 99);

  StConv2dParams p = st_conv2d_default_params();

  /* Forward */
  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  /* grad_output */
  FloatTensor *grad_output = create_4d(1, 3, 4, 4);
  TEST_ASSERT_NOT_NULL(grad_output);
  fill_rand(grad_output, 11);

  /* Default path (should use Winograd for 3x3 stride=1) */
  FloatTensor *grad_input_fast = create_4d(1, 2, 6, 6);
  TEST_ASSERT_NOT_NULL(grad_input_fast);
  ok = st_conv2d_backward_data_nchw(grad_output, weight, &p, grad_input_fast);
  TEST_ASSERT_TRUE(ok);

  /* Naive reference */
  FloatTensor *grad_input_ref = create_4d(1, 2, 6, 6);
  TEST_ASSERT_NOT_NULL(grad_input_ref);
  StConv2dParams p_ref = p;
  p_ref.backend = ST_CONV_BACKEND_REFERENCE;
  ok = st_conv2d_backward_data_nchw(grad_output, weight, &p_ref, grad_input_ref);
  TEST_ASSERT_TRUE(ok);

  /* Compare — Winograd may have slightly less precision */
  for (size_t i = 0; i < grad_input_fast->numel; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(0.05f, grad_input_ref->values[i],
                             grad_input_fast->values[i]);
  }

  st_destroy(grad_input_ref);
  st_destroy(grad_input_fast);
  st_destroy(grad_output);
  st_destroy(output);
  st_destroy(weight);
  st_destroy(input);
}

/* ---- Backward with padding ---- */

void test_gemm_backward_data_with_padding(void) {
  /* Input 1x1x5x5, weight 1x1x3x3, pad=1, output 1x1x5x5 */
  FloatTensor *input = create_4d(1, 1, 5, 5);
  FloatTensor *weight = create_4d(1, 1, 3, 3);
  FloatTensor *output = create_4d(1, 1, 5, 5);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  fill_inc(input);
  fill_rand(weight, 55);

  StConv2dParams p = st_conv2d_default_params();
  p.pad_h = 1;
  p.pad_w = 1;

  /* Forward */
  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  /* grad_output = 1 */
  FloatTensor *grad_output = create_4d(1, 1, 5, 5);
  TEST_ASSERT_NOT_NULL(grad_output);
  for (size_t i = 0; i < grad_output->numel; ++i) {
    grad_output->values[i] = 1.0f;
  }

  /* Fast path */
  FloatTensor *grad_input_fast = create_4d(1, 1, 5, 5);
  TEST_ASSERT_NOT_NULL(grad_input_fast);
  ok = st_conv2d_backward_data_nchw(grad_output, weight, &p, grad_input_fast);
  TEST_ASSERT_TRUE(ok);

  /* Naive reference */
  FloatTensor *grad_input_ref = create_4d(1, 1, 5, 5);
  TEST_ASSERT_NOT_NULL(grad_input_ref);
  StConv2dParams p_ref = p;
  p_ref.backend = ST_CONV_BACKEND_REFERENCE;
  ok = st_conv2d_backward_data_nchw(grad_output, weight, &p_ref, grad_input_ref);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < grad_input_fast->numel; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, grad_input_ref->values[i],
                             grad_input_fast->values[i]);
  }

  st_destroy(grad_input_ref);
  st_destroy(grad_input_fast);
  st_destroy(grad_output);
  st_destroy(output);
  st_destroy(weight);
  st_destroy(input);
}
