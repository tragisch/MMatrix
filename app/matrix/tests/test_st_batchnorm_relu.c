/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_batchnorm.h"

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

/* Fill tensor with pseudo-random values in [-1, 1]. */
static void fill_rand(FloatTensor *t, unsigned int seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1103515245u + 12345u;
    t->values[i] = ((float)(seed >> 16) / 32768.0f) - 1.0f;
  }
}

/* ---- Fused BN+ReLU forward matches separate BN then ReLU ---- */

void test_fused_bn_relu_forward_matches_separate(void) {
  FloatTensor *input = create_4d(2, 3, 4, 4);
  FloatTensor *gamma = create_1d(3);
  FloatTensor *beta = create_1d(3);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(gamma);
  TEST_ASSERT_NOT_NULL(beta);

  fill_rand(input, 42);
  for (size_t i = 0; i < 3; ++i) {
    gamma->values[i] = 1.0f + 0.1f * (float)i;
    beta->values[i] = -0.5f + 0.3f * (float)i;
  }

  const float eps = 1e-5f;

  /* Separate: BN then ReLU */
  FloatTensor *output_sep = create_4d(2, 3, 4, 4);
  FloatTensor *mean_sep = create_1d(3);
  FloatTensor *var_sep = create_1d(3);
  TEST_ASSERT_NOT_NULL(output_sep);
  TEST_ASSERT_NOT_NULL(mean_sep);
  TEST_ASSERT_NOT_NULL(var_sep);

  bool ok = st_batchnorm2d_forward(input, gamma, beta, eps, output_sep,
                                   mean_sep, var_sep);
  TEST_ASSERT_TRUE(ok);
  ok = st_apply_relu(output_sep);
  TEST_ASSERT_TRUE(ok);

  /* Fused */
  FloatTensor *output_fused = create_4d(2, 3, 4, 4);
  FloatTensor *mean_fused = create_1d(3);
  FloatTensor *var_fused = create_1d(3);
  TEST_ASSERT_NOT_NULL(output_fused);
  TEST_ASSERT_NOT_NULL(mean_fused);
  TEST_ASSERT_NOT_NULL(var_fused);

  ok = st_batchnorm2d_forward_relu(input, gamma, beta, eps, output_fused,
                                   mean_fused, var_fused);
  TEST_ASSERT_TRUE(ok);

  /* Compare outputs */
  for (size_t i = 0; i < output_sep->numel; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, output_sep->values[i],
                             output_fused->values[i]);
  }

  /* Compare mean/var */
  for (size_t i = 0; i < 3; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, mean_sep->values[i],
                             mean_fused->values[i]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, var_sep->values[i],
                             var_fused->values[i]);
  }

  st_destroy(var_fused);
  st_destroy(mean_fused);
  st_destroy(output_fused);
  st_destroy(var_sep);
  st_destroy(mean_sep);
  st_destroy(output_sep);
  st_destroy(beta);
  st_destroy(gamma);
  st_destroy(input);
}

/* ---- Fused BN+ReLU forward with NULL gamma/beta ---- */

void test_fused_bn_relu_forward_null_gamma_beta(void) {
  FloatTensor *input = create_4d(1, 2, 3, 3);
  TEST_ASSERT_NOT_NULL(input);
  fill_rand(input, 99);

  const float eps = 1e-5f;

  FloatTensor *output_sep = create_4d(1, 2, 3, 3);
  FloatTensor *mean_sep = create_1d(2);
  FloatTensor *var_sep = create_1d(2);
  TEST_ASSERT_NOT_NULL(output_sep);

  bool ok = st_batchnorm2d_forward(input, NULL, NULL, eps, output_sep,
                                   mean_sep, var_sep);
  TEST_ASSERT_TRUE(ok);
  ok = st_apply_relu(output_sep);
  TEST_ASSERT_TRUE(ok);

  FloatTensor *output_fused = create_4d(1, 2, 3, 3);
  FloatTensor *mean_fused = create_1d(2);
  FloatTensor *var_fused = create_1d(2);
  TEST_ASSERT_NOT_NULL(output_fused);

  ok = st_batchnorm2d_forward_relu(input, NULL, NULL, eps, output_fused,
                                   mean_fused, var_fused);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < output_sep->numel; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, output_sep->values[i],
                             output_fused->values[i]);
  }

  st_destroy(var_fused);
  st_destroy(mean_fused);
  st_destroy(output_fused);
  st_destroy(var_sep);
  st_destroy(mean_sep);
  st_destroy(output_sep);
  st_destroy(input);
}

/* ---- Fused BN+ReLU forward: all negatives after BN should be zero ---- */

void test_fused_bn_relu_forward_all_negative_clipped(void) {
  /* Create input where after BN all values should be negative. 
   * Set beta to a large negative value. */
  FloatTensor *input = create_4d(1, 1, 2, 2);
  FloatTensor *gamma = create_1d(1);
  FloatTensor *beta = create_1d(1);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(gamma);
  TEST_ASSERT_NOT_NULL(beta);

  /* All same value → zero variance → normalized to 0; beta=-10 makes it all <0 */
  for (size_t i = 0; i < 4; ++i) input->values[i] = 5.0f;
  gamma->values[0] = 1.0f;
  beta->values[0] = -10.0f;

  FloatTensor *output = create_4d(1, 1, 2, 2);
  FloatTensor *mean = create_1d(1);
  FloatTensor *var = create_1d(1);
  TEST_ASSERT_NOT_NULL(output);

  bool ok = st_batchnorm2d_forward_relu(input, gamma, beta, 1e-5f, output,
                                        mean, var);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 4; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, output->values[i]);
  }

  st_destroy(var);
  st_destroy(mean);
  st_destroy(output);
  st_destroy(beta);
  st_destroy(gamma);
  st_destroy(input);
}

/* ---- Fused BN+ReLU backward matches separate ReLU-backward + BN-backward ---- */

void test_fused_bn_relu_backward_matches_separate(void) {
  FloatTensor *input = create_4d(2, 3, 4, 4);
  FloatTensor *gamma = create_1d(3);
  FloatTensor *beta = create_1d(3);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(gamma);
  TEST_ASSERT_NOT_NULL(beta);

  fill_rand(input, 42);
  for (size_t i = 0; i < 3; ++i) {
    gamma->values[i] = 1.0f + 0.1f * (float)i;
    beta->values[i] = -0.1f + 0.2f * (float)i;
  }

  const float eps = 1e-5f;

  /* Fused forward */
  FloatTensor *bn_output = create_4d(2, 3, 4, 4);
  FloatTensor *mean = create_1d(3);
  FloatTensor *var = create_1d(3);
  TEST_ASSERT_NOT_NULL(bn_output);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);

  bool ok = st_batchnorm2d_forward_relu(input, gamma, beta, eps, bn_output,
                                        mean, var);
  TEST_ASSERT_TRUE(ok);

  /* grad_output */
  FloatTensor *grad_output = create_4d(2, 3, 4, 4);
  TEST_ASSERT_NOT_NULL(grad_output);
  fill_rand(grad_output, 77);

  /* Fused backward */
  FloatTensor *grad_input_fused = create_4d(2, 3, 4, 4);
  FloatTensor *grad_gamma_fused = create_1d(3);
  FloatTensor *grad_beta_fused = create_1d(3);
  TEST_ASSERT_NOT_NULL(grad_input_fused);
  TEST_ASSERT_NOT_NULL(grad_gamma_fused);
  TEST_ASSERT_NOT_NULL(grad_beta_fused);

  ok = st_batchnorm2d_backward_relu(grad_output, input, bn_output, mean, var,
                                    gamma, eps, grad_input_fused,
                                    grad_gamma_fused, grad_beta_fused);
  TEST_ASSERT_TRUE(ok);

  /* Separate backward: manually apply ReLU mask to grad_output, then BN backward */
  FloatTensor *grad_output_masked = create_4d(2, 3, 4, 4);
  TEST_ASSERT_NOT_NULL(grad_output_masked);
  for (size_t i = 0; i < grad_output->numel; ++i) {
    grad_output_masked->values[i] =
        (bn_output->values[i] > 0.0f) ? grad_output->values[i] : 0.0f;
  }

  FloatTensor *grad_input_sep = create_4d(2, 3, 4, 4);
  FloatTensor *grad_gamma_sep = create_1d(3);
  FloatTensor *grad_beta_sep = create_1d(3);
  TEST_ASSERT_NOT_NULL(grad_input_sep);
  TEST_ASSERT_NOT_NULL(grad_gamma_sep);
  TEST_ASSERT_NOT_NULL(grad_beta_sep);

  ok = st_batchnorm2d_backward(grad_output_masked, input, mean, var, gamma,
                               eps, grad_input_sep, grad_gamma_sep,
                               grad_beta_sep);
  TEST_ASSERT_TRUE(ok);

  /* Compare grad_input */
  for (size_t i = 0; i < grad_input_fused->numel; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, grad_input_sep->values[i],
                             grad_input_fused->values[i]);
  }

  /* Compare grad_gamma and grad_beta */
  for (size_t i = 0; i < 3; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, grad_gamma_sep->values[i],
                             grad_gamma_fused->values[i]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, grad_beta_sep->values[i],
                             grad_beta_fused->values[i]);
  }

  st_destroy(grad_beta_sep);
  st_destroy(grad_gamma_sep);
  st_destroy(grad_input_sep);
  st_destroy(grad_output_masked);
  st_destroy(grad_beta_fused);
  st_destroy(grad_gamma_fused);
  st_destroy(grad_input_fused);
  st_destroy(grad_output);
  st_destroy(var);
  st_destroy(mean);
  st_destroy(bn_output);
  st_destroy(beta);
  st_destroy(gamma);
  st_destroy(input);
}

/* ---- Fused BN+ReLU backward with NULL grad_gamma/grad_beta ---- */

void test_fused_bn_relu_backward_null_grads(void) {
  FloatTensor *input = create_4d(1, 2, 3, 3);
  FloatTensor *gamma = create_1d(2);
  FloatTensor *beta = create_1d(2);
  TEST_ASSERT_NOT_NULL(input);
  fill_rand(input, 99);
  gamma->values[0] = 1.0f;
  gamma->values[1] = 0.5f;
  beta->values[0] = 0.0f;
  beta->values[1] = 0.1f;

  const float eps = 1e-5f;

  FloatTensor *bn_output = create_4d(1, 2, 3, 3);
  FloatTensor *mean = create_1d(2);
  FloatTensor *var = create_1d(2);
  TEST_ASSERT_NOT_NULL(bn_output);

  bool ok = st_batchnorm2d_forward_relu(input, gamma, beta, eps, bn_output,
                                        mean, var);
  TEST_ASSERT_TRUE(ok);

  FloatTensor *grad_output = create_4d(1, 2, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_output);
  fill_rand(grad_output, 55);

  FloatTensor *grad_input = create_4d(1, 2, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_input);

  /* Should work with NULL grad_gamma and grad_beta */
  ok = st_batchnorm2d_backward_relu(grad_output, input, bn_output, mean, var,
                                    gamma, eps, grad_input, NULL, NULL);
  TEST_ASSERT_TRUE(ok);

  /* Just check it didn't crash and grad_input has reasonable values */
  bool has_nonzero = false;
  for (size_t i = 0; i < grad_input->numel; ++i) {
    if (grad_input->values[i] != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  TEST_ASSERT_TRUE(has_nonzero);

  st_destroy(grad_input);
  st_destroy(grad_output);
  st_destroy(var);
  st_destroy(mean);
  st_destroy(bn_output);
  st_destroy(beta);
  st_destroy(gamma);
  st_destroy(input);
}
