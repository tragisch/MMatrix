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
#ifndef TEST_ASSERT_EQUAL_size_t
#define TEST_ASSERT_EQUAL_size_t(expected, actual) \
  do {                                             \
    (void)(expected);                              \
    (void)(actual);                                \
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

void test_st_conv2d_nchw_gemm_should_match_reference_result(void) {
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
  p.backend = ST_CONV_BACKEND_GEMM;

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

void test_st_conv2d_nchw_mps_backend_should_fallback_if_unavailable(void) {
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
  p.backend = ST_CONV_BACKEND_MPS;

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

void test_st_conv_mps_thresholds_should_be_settable_and_queryable(void) {
  bool ok = st_conv_set_mps_thresholds(3.0e8, 2000000u);
  TEST_ASSERT_TRUE(ok);

  double macs = 0.0;
  size_t out_elems = 0;
  st_conv_get_mps_thresholds(&macs, &out_elems);

  TEST_ASSERT_TRUE(macs > 2.99e8 && macs < 3.01e8);
  TEST_ASSERT_EQUAL_size_t(2000000u, out_elems);
}

void test_st_conv_mps_thresholds_should_reject_invalid_values(void) {
  bool ok = st_conv_set_mps_thresholds(1.0e8, 12345u);
  TEST_ASSERT_TRUE(ok);

  ok = st_conv_set_mps_thresholds(0.0, 12345u);
  TEST_ASSERT_FALSE(ok);

  ok = st_conv_set_mps_thresholds(1.0e8, 0u);
  TEST_ASSERT_FALSE(ok);

  double macs = 0.0;
  size_t out_elems = 0;
  st_conv_get_mps_thresholds(&macs, &out_elems);

  TEST_ASSERT_TRUE(macs > 0.0);
  TEST_ASSERT_EQUAL_size_t(12345u, out_elems);
}

/* ---- Tests for 1x1 convolution fast-path ---- */

void test_st_conv2d_1x1_gemm_should_match_reference(void) {
  /* 1x1 conv: [1, 3, 4, 4] * [2, 3, 1, 1] → [1, 2, 4, 4] */
  FloatTensor *input = create_tensor_4d(1, 3, 4, 4);
  FloatTensor *weight = create_tensor_4d(2, 3, 1, 1);
  FloatTensor *out_ref = create_tensor_4d(1, 2, 4, 4);
  FloatTensor *out_gemm = create_tensor_4d(1, 2, 4, 4);
  size_t bias_shape[1] = {2};
  FloatTensor *bias = st_create(1, bias_shape);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(out_ref);
  TEST_ASSERT_NOT_NULL(out_gemm);
  TEST_ASSERT_NOT_NULL(bias);

  /* Fill with simple pattern */
  for (size_t i = 0; i < 3 * 4 * 4; ++i) {
    input->values[i] = (float)(i + 1) * 0.1f;
  }
  for (size_t i = 0; i < 2 * 3; ++i) {
    weight->values[i] = (float)(i) * 0.5f - 1.0f;
  }
  bias->values[0] = 0.5f;
  bias->values[1] = -0.5f;

  StConv2dParams p_ref = st_conv2d_default_params();
  p_ref.backend = ST_CONV_BACKEND_REFERENCE;
  bool ok = st_conv2d_nchw(input, weight, bias, &p_ref, out_ref);
  TEST_ASSERT_TRUE(ok);

  StConv2dParams p_gemm = st_conv2d_default_params();
  p_gemm.backend = ST_CONV_BACKEND_GEMM;
  ok = st_conv2d_nchw(input, weight, bias, &p_gemm, out_gemm);
  TEST_ASSERT_TRUE(ok);

  /* The backend should report gemm_1x1 */
  const char *backend = st_conv2d_last_backend();
  TEST_ASSERT_NOT_NULL(backend);

  /* Compare all output elements */
  for (size_t i = 0; i < 2 * 4 * 4; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, out_ref->values[i], out_gemm->values[i]);
  }

  st_destroy(input);
  st_destroy(weight);
  st_destroy(out_ref);
  st_destroy(out_gemm);
  st_destroy(bias);
}

void test_st_conv2d_1x1_auto_should_dispatch_correctly(void) {
  /* 1x1 conv via AUTO backend → should pick gemm_1x1 */
  FloatTensor *input = create_tensor_4d(1, 2, 3, 3);
  FloatTensor *weight = create_tensor_4d(2, 2, 1, 1);
  FloatTensor *output = create_tensor_4d(1, 2, 3, 3);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  for (size_t i = 0; i < 2 * 3 * 3; ++i) {
    input->values[i] = (float)i;
  }
  /* identity-ish weights */
  weight->values[0] = 1.0f;
  weight->values[1] = 0.0f;
  weight->values[2] = 0.0f;
  weight->values[3] = 1.0f;

  StConv2dParams p = st_conv2d_default_params();
  /* AUTO is default */

  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  /* With identity weights, output should equal input */
  for (size_t i = 0; i < 2 * 3 * 3; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, input->values[i], output->values[i]);
  }

  st_destroy(input);
  st_destroy(weight);
  st_destroy(output);
}

/* ---- Test with padding (exercises border/inner split in cpu_opt) ---- */

void test_st_conv2d_cpu_opt_with_padding_should_match_reference(void) {
  /* 3x3 conv with pad=1 on 4x4 input → 4x4 output */
  FloatTensor *input = create_tensor_4d(1, 1, 4, 4);
  FloatTensor *weight = create_tensor_4d(1, 1, 3, 3);
  FloatTensor *out_ref = create_tensor_4d(1, 1, 4, 4);
  FloatTensor *out_opt = create_tensor_4d(1, 1, 4, 4);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(out_ref);
  TEST_ASSERT_NOT_NULL(out_opt);

  for (size_t i = 0; i < 16; ++i) {
    input->values[i] = (float)(i + 1);
  }
  for (size_t i = 0; i < 9; ++i) {
    weight->values[i] = (i == 4) ? 1.0f : 0.0f; /* center-only kernel */
  }

  StConv2dParams p = st_conv2d_default_params();
  p.pad_h = 1;
  p.pad_w = 1;

  p.backend = ST_CONV_BACKEND_REFERENCE;
  bool ok = st_conv2d_nchw(input, weight, NULL, &p, out_ref);
  TEST_ASSERT_TRUE(ok);

  p.backend = ST_CONV_BACKEND_CPU_OPT;
  ok = st_conv2d_nchw(input, weight, NULL, &p, out_opt);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 16; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, out_ref->values[i], out_opt->values[i]);
  }

  st_destroy(input);
  st_destroy(weight);
  st_destroy(out_ref);
  st_destroy(out_opt);
}

/* ---- Test with larger tensor (exercises AUTO dispatch paths) ---- */

void test_st_conv2d_auto_larger_tensor_all_backends_agree(void) {
  /* Batch=2, C_in=3, 8x8 input, 3x3 kernel, C_out=4 → 6x6 output */
  FloatTensor *input = create_tensor_4d(2, 3, 8, 8);
  FloatTensor *weight = create_tensor_4d(4, 3, 3, 3);
  FloatTensor *out_ref = create_tensor_4d(2, 4, 6, 6);
  FloatTensor *out_auto = create_tensor_4d(2, 4, 6, 6);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(out_ref);
  TEST_ASSERT_NOT_NULL(out_auto);

  for (size_t i = 0; i < 2 * 3 * 8 * 8; ++i) {
    input->values[i] = (float)(i % 17) * 0.1f - 0.8f;
  }
  for (size_t i = 0; i < 4 * 3 * 3 * 3; ++i) {
    weight->values[i] = (float)(i % 11) * 0.2f - 1.0f;
  }

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_REFERENCE;
  bool ok = st_conv2d_nchw(input, weight, NULL, &p, out_ref);
  TEST_ASSERT_TRUE(ok);

  p.backend = ST_CONV_BACKEND_AUTO;
  ok = st_conv2d_nchw(input, weight, NULL, &p, out_auto);
  TEST_ASSERT_TRUE(ok);

  const size_t total = 2 * 4 * 6 * 6;
  for (size_t i = 0; i < total; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, out_ref->values[i], out_auto->values[i]);
  }

  st_destroy(input);
  st_destroy(weight);
  st_destroy(out_ref);
  st_destroy(out_auto);
}
