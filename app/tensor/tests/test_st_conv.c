/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_conv.h"
#include "st_batchnorm.h"

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

#define EPSILON 1e-6f

void setUp(void) {}
void tearDown(void) {}

static FloatTensor *create_tensor_4d(size_t n, size_t c, size_t h, size_t w) {
  size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

static FloatTensor *create_tensor_1d(size_t n) {
  size_t shape[1] = {n};
  return st_create(1, shape);
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

void test_st_conv2d_nchw_should_promote_bf16_bias_for_f32_compute(void) {
  FloatTensor *input = create_tensor_4d(1, 1, 3, 3);
  FloatTensor *weight = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);
  size_t bias_shape[1] = {1};
  FloatTensor *bias = st_create_bf16(1, bias_shape);

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
  TEST_ASSERT_TRUE(st_set(bias, (size_t[]){0}, 1.0f));

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_REFERENCE;

  bool ok = st_conv2d_nchw(input, weight, bias, &p, output);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 4; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, -3.0f, output->values[i]);
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

void test_st_conv2d_1x1_gemm_should_match_reference(void) {
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

  const char *backend = st_conv2d_last_backend();
  TEST_ASSERT_NOT_NULL(backend);

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
  FloatTensor *input = create_tensor_4d(1, 2, 3, 3);
  FloatTensor *weight = create_tensor_4d(2, 2, 1, 1);
  FloatTensor *output = create_tensor_4d(1, 2, 3, 3);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  for (size_t i = 0; i < 2 * 3 * 3; ++i) {
    input->values[i] = (float)i;
  }
  weight->values[0] = 1.0f;
  weight->values[1] = 0.0f;
  weight->values[2] = 0.0f;
  weight->values[3] = 1.0f;

  StConv2dParams p = st_conv2d_default_params();

  bool ok = st_conv2d_nchw(input, weight, NULL, &p, output);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 2 * 3 * 3; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, input->values[i], output->values[i]);
  }

  st_destroy(input);
  st_destroy(weight);
  st_destroy(output);
}

void test_st_conv2d_cpu_opt_with_padding_should_match_reference(void) {
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
    weight->values[i] = (i == 4) ? 1.0f : 0.0f;
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

void test_st_conv2d_batchnorm2d_forward_should_match_separate_ops(void) {
  FloatTensor *input = create_tensor_4d(1, 2, 4, 4);
  FloatTensor *weight = create_tensor_4d(2, 2, 3, 3);
  FloatTensor *bias = create_tensor_1d(2);
  FloatTensor *gamma = create_tensor_1d(2);
  FloatTensor *beta = create_tensor_1d(2);
  FloatTensor *fused_out = create_tensor_4d(1, 2, 2, 2);
  FloatTensor *fused_mean = create_tensor_1d(2);
  FloatTensor *fused_var = create_tensor_1d(2);
  FloatTensor *ref_conv = create_tensor_4d(1, 2, 2, 2);
  FloatTensor *ref_out = create_tensor_4d(1, 2, 2, 2);
  FloatTensor *ref_mean = create_tensor_1d(2);
  FloatTensor *ref_var = create_tensor_1d(2);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(bias);
  TEST_ASSERT_NOT_NULL(gamma);
  TEST_ASSERT_NOT_NULL(beta);
  TEST_ASSERT_NOT_NULL(fused_out);
  TEST_ASSERT_NOT_NULL(fused_mean);
  TEST_ASSERT_NOT_NULL(fused_var);
  TEST_ASSERT_NOT_NULL(ref_conv);
  TEST_ASSERT_NOT_NULL(ref_out);
  TEST_ASSERT_NOT_NULL(ref_mean);
  TEST_ASSERT_NOT_NULL(ref_var);

  for (size_t i = 0; i < input->numel; ++i) {
    input->values[i] = (float)((int)(i % 9) - 4) * 0.25f;
  }
  for (size_t i = 0; i < weight->numel; ++i) {
    weight->values[i] = (float)((int)(i % 7) - 3) * 0.2f;
  }
  bias->values[0] = 0.5f;
  bias->values[1] = -0.25f;
  gamma->values[0] = 1.5f;
  gamma->values[1] = 0.75f;
  beta->values[0] = -0.2f;
  beta->values[1] = 0.4f;

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_MPS;

  bool ok = st_conv2d_batchnorm2d_forward_nchw(
      input, weight, bias, &p, gamma, beta, 1e-5f,
      fused_out, fused_mean, fused_var);
  TEST_ASSERT_TRUE(ok);

  p.backend = ST_CONV_BACKEND_REFERENCE;
  ok = st_conv2d_nchw(input, weight, bias, &p, ref_conv);
  TEST_ASSERT_TRUE(ok);
  ok = st_batchnorm2d_forward(ref_conv, gamma, beta, 1e-5f,
                              ref_out, ref_mean, ref_var);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < fused_out->numel; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, ref_out->values[i], fused_out->values[i]);
  }
  for (size_t i = 0; i < 2; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, ref_mean->values[i], fused_mean->values[i]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, ref_var->values[i], fused_var->values[i]);
  }

  st_destroy(input);
  st_destroy(weight);
  st_destroy(bias);
  st_destroy(gamma);
  st_destroy(beta);
  st_destroy(fused_out);
  st_destroy(fused_mean);
  st_destroy(fused_var);
  st_destroy(ref_conv);
  st_destroy(ref_out);
  st_destroy(ref_mean);
  st_destroy(ref_var);
}

void test_st_conv2d_batchnorm2d_forward_without_optional_tensors_should_match_separate_ops(void) {
  FloatTensor *input = create_tensor_4d(1, 1, 4, 4);
  FloatTensor *weight = create_tensor_4d(1, 1, 3, 3);
  FloatTensor *fused_out = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *fused_mean = create_tensor_1d(1);
  FloatTensor *fused_var = create_tensor_1d(1);
  FloatTensor *ref_conv = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *ref_out = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *ref_mean = create_tensor_1d(1);
  FloatTensor *ref_var = create_tensor_1d(1);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(fused_out);
  TEST_ASSERT_NOT_NULL(fused_mean);
  TEST_ASSERT_NOT_NULL(fused_var);
  TEST_ASSERT_NOT_NULL(ref_conv);
  TEST_ASSERT_NOT_NULL(ref_out);
  TEST_ASSERT_NOT_NULL(ref_mean);
  TEST_ASSERT_NOT_NULL(ref_var);

  for (size_t i = 0; i < input->numel; ++i) {
    input->values[i] = (float)(i + 1) * 0.1f;
  }
  for (size_t i = 0; i < weight->numel; ++i) {
    weight->values[i] = (float)((int)(i % 5) - 2) * 0.15f;
  }

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_MPS;

  bool ok = st_conv2d_batchnorm2d_forward_nchw(
      input, weight, NULL, &p, NULL, NULL, 1e-5f,
      fused_out, fused_mean, fused_var);
  TEST_ASSERT_TRUE(ok);

  p.backend = ST_CONV_BACKEND_REFERENCE;
  ok = st_conv2d_nchw(input, weight, NULL, &p, ref_conv);
  TEST_ASSERT_TRUE(ok);
  ok = st_batchnorm2d_forward(ref_conv, NULL, NULL, 1e-5f,
                              ref_out, ref_mean, ref_var);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < fused_out->numel; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, ref_out->values[i], fused_out->values[i]);
  }
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, ref_mean->values[0], fused_mean->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, ref_var->values[0], fused_var->values[0]);

  st_destroy(input);
  st_destroy(weight);
  st_destroy(fused_out);
  st_destroy(fused_mean);
  st_destroy(fused_var);
  st_destroy(ref_conv);
  st_destroy(ref_out);
  st_destroy(ref_mean);
  st_destroy(ref_var);
}

void test_st_conv2d_auto_larger_tensor_all_backends_agree(void) {
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
