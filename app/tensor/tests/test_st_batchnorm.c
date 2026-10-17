/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_batchnorm.h"
#include "st.h"
#include "st_bf16_utils.h"

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

void test_st_batchnorm2d_forward_uniform_input_should_output_zero(void) {
  FloatTensor *input = create_4d(2, 1, 2, 2);
  FloatTensor *output = create_4d(2, 1, 2, 2);
  FloatTensor *mean = create_1d(1);
  FloatTensor *var = create_1d(1);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);

  for (size_t i = 0; i < 8; ++i) {
    input->values[i] = 5.0f;
  }

  bool ok = st_batchnorm2d_forward(input, NULL, NULL, 1e-5f, output, mean, var);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, mean->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, var->values[0]);

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

void test_st_batchnorm2d_backward_gradient_consistency(void) {
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

  bool ok = st_batchnorm2d_forward(input, NULL, NULL, eps, output, mean, var);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 4; ++i) {
    grad_output->values[i] = 1.0f;
  }

  ok = st_batchnorm2d_backward(grad_output, input, mean, var, NULL, eps,
                               grad_input, NULL, NULL);
  TEST_ASSERT_TRUE(ok);

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

/* ------------------------------------------------------------------ */
/*  BF16 promotion test (Item #8 — BF16 policy)                       */
/* ------------------------------------------------------------------ */

/*
 * Verify that st_batchnorm2d_forward auto-promotes BF16 input/output
 * and gamma/beta tensors to F32 for computation.
 *
 * Input: 2×2×2×2 (N=2, C=2, H=2, W=2) with alternating +1 / -1 (BF16).
 * gamma=1, beta=0 (F32 default).
 * After BN: output should be normalised; all values within F32 tolerance.
 */
void test_st_batchnorm2d_bf16_promotion(void) {
  /* N=2, C=2, H=2, W=2 → numel=16 */
  size_t shape4[4] = {2, 2, 2, 2};
  size_t shape1[1] = {2};
  const size_t numel = 16;

  FloatTensor *input  = st_create_bf16(4, shape4);
  FloatTensor *output = st_create_bf16(4, shape4);
  FloatTensor *mean   = st_create(1, shape1);
  FloatTensor *var    = st_create(1, shape1);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);

  /* Fill alternating +1 / -1 in BF16 (numel elements, not 32) */
  for (size_t i = 0; i < numel; ++i)
    ((uint16_t *)input->values)[i] = st_f32_to_bf16(i % 2 == 0 ? 1.0f : -1.0f);

  bool ok = st_batchnorm2d_forward(input, NULL, NULL, 1e-5f, output, mean, var);
  TEST_ASSERT_TRUE(ok);

  /* Mean of ±1 alternating sequence is ~0; output should be normalised near ±1 */
  for (size_t i = 0; i < numel; ++i) {
    float v = st_bf16_to_f32(((uint16_t *)output->values)[i]);
    TEST_ASSERT_FALSE(v != v);  /* no NaN */
    TEST_ASSERT_FLOAT_WITHIN(0.1f, (i % 2 == 0 ? 1.0f : -1.0f), v);
  }

  st_destroy(input);
  st_destroy(output);
  st_destroy(mean);
  st_destroy(var);
}
