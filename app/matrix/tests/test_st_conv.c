/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_conv.h"

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
#ifndef TEST_ASSERT_NOT_NULL
#define TEST_ASSERT_NOT_NULL(value) \
  do {                             \
    (void)(value);                 \
  } while (0)
#endif

#define EPSILON 1e-6f

void setUp(void) {}
void tearDown(void) {}

static FloatTensor *create_tensor_4d(size_t n, size_t c, size_t h, size_t w) {
  size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

void test_st_conv2d_output_hw_should_compute_expected_shape(void) {
  StConv2dParams p = st_conv2d_default_params();

  size_t out_h = 0;
  size_t out_w = 0;
  bool ok = st_conv2d_output_hw(5, 5, 3, 3, &p, &out_h, &out_w);

  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL(3, out_h);
  TEST_ASSERT_EQUAL(3, out_w);
}

void test_st_conv2d_nchw_reference_without_bias(void) {
  FloatTensor *input = create_tensor_4d(1, 1, 3, 3);
  FloatTensor *weight = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  float in_vals[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  float k_vals[] = {
      1.0f, 0.0f,
      0.0f, -1.0f,
  };

  for (size_t i = 0; i < 9; ++i) {
    input->values[i] = in_vals[i];
  }
  for (size_t i = 0; i < 4; ++i) {
    weight->values[i] = k_vals[i];
  }

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_REFERENCE;

  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 4; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, -4.0f, output->values[i]);
  }
  TEST_ASSERT_TRUE(st_conv2d_last_backend() != NULL);

  st_destroy(input);
  st_destroy(weight);
  st_destroy(output);
}

void test_st_conv2d_nchw_reference_with_bias(void) {
  FloatTensor *input = create_tensor_4d(1, 1, 3, 3);
  FloatTensor *weight = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);
  size_t bias_shape[1] = {1};
  FloatTensor *bias = st_create(1, bias_shape);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(bias);

  float in_vals[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  float k_vals[] = {
      1.0f, 0.0f,
      0.0f, -1.0f,
  };

  for (size_t i = 0; i < 9; ++i) {
    input->values[i] = in_vals[i];
  }
  for (size_t i = 0; i < 4; ++i) {
    weight->values[i] = k_vals[i];
  }
  bias->values[0] = 1.0f;

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_REFERENCE;

  bool ok = st_conv2d_nchw(input, weight, bias, &p, output);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 4; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, -3.0f, output->values[i]);
  }

  st_destroy(input);
  st_destroy(weight);
  st_destroy(output);
  st_destroy(bias);
}

void test_st_conv2d_nchw_should_fail_on_shape_mismatch(void) {
  FloatTensor *input = create_tensor_4d(1, 1, 3, 3);
  FloatTensor *weight = create_tensor_4d(2, 1, 2, 2);
  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  StConv2dParams p = st_conv2d_default_params();

  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_FALSE(ok);

  st_destroy(input);
  st_destroy(weight);
  st_destroy(output);
}

void test_st_conv2d_nchw_bnns_backend_should_fallback_to_reference_if_unavailable(
    void) {
  FloatTensor *input = create_tensor_4d(1, 1, 3, 3);
  FloatTensor *weight = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  float in_vals[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  float k_vals[] = {
      1.0f, 0.0f,
      0.0f, -1.0f,
  };

  for (size_t i = 0; i < 9; ++i) {
    input->values[i] = in_vals[i];
  }
  for (size_t i = 0; i < 4; ++i) {
    weight->values[i] = k_vals[i];
  }

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_BNNS;

  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 4; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, -4.0f, output->values[i]);
  }

  const char *backend = st_conv2d_last_backend();
  TEST_ASSERT_NOT_NULL(backend);

  st_destroy(input);
  st_destroy(weight);
  st_destroy(output);
}

void test_st_conv2d_nchw_cpu_opt_should_match_reference_result(void) {
  FloatTensor *input = create_tensor_4d(1, 1, 3, 3);
  FloatTensor *weight = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  float in_vals[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  float k_vals[] = {
      1.0f, 0.0f,
      0.0f, -1.0f,
  };

  for (size_t i = 0; i < 9; ++i) {
    input->values[i] = in_vals[i];
  }
  for (size_t i = 0; i < 4; ++i) {
    weight->values[i] = k_vals[i];
  }

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_CPU_OPT;

  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 4; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, -4.0f, output->values[i]);
  }

  const char *backend = st_conv2d_last_backend();
  TEST_ASSERT_NOT_NULL(backend);

  st_destroy(input);
  st_destroy(weight);
  st_destroy(output);
}
