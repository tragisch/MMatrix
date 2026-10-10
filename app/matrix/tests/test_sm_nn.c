#include "sm.h"
#include "sm_kernels.h"

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 6
#define TEST_CASE(...)
#include "unity.h"

static bool near_value(float actual, float expected) {
  return fabsf(actual - expected) <= 1.0e-6f + 1.0e-5f * fabsf(expected);
}

static float reference_gelu_tanh(float x) {
  const float k = 0.7978845608f;
  return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

static float reference_gelu_erf(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.7071067812f));
}

static float reference_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

static float reference_activation(SmActivation activation, float x) {
  switch (activation) {
    case SM_ACT_NONE: return x;
    case SM_ACT_RELU: return x > 0.0f ? x : 0.0f;
    case SM_ACT_SIGMOID: return 1.0f / (1.0f + expf(-x));
    case SM_ACT_TANH: return tanhf(x);
  }
  return NAN;
}

void setUp(void) {}
void tearDown(void) {}

void test_sm_activation_kernels_match_scalar_reference_sizes(void) {
  const size_t sizes[] = {1, 7, 1000, 100000};
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
    const size_t n = sizes[s];
    FloatMatrix *mat = sm_create(1, n);
    TEST_ASSERT_NOT_NULL(mat);
    uint32_t random_state = 0x12345678u;
    for (size_t i = 0; i < n; ++i) {
      random_state = random_state * 1664525u + 1013904223u;
      mat->values[i] = ((float)(random_state >> 8) / 16777216.0f - 0.5f) * 8.0f;
    }

    FloatMatrix *actual = sm_clone(mat);
    TEST_ASSERT_TRUE(sm_apply_sigmoid(actual));
    for (size_t i = 0; i < n; ++i) {
      const float x = mat->values[i];
      TEST_ASSERT_TRUE(near_value(actual->values[i],
                                  1.0f / (1.0f + expf(-x))));
    }
    sm_destroy(actual);

    actual = sm_clone(mat);
    TEST_ASSERT_TRUE(sm_apply_tanh(actual));
    for (size_t i = 0; i < n; ++i) {
      TEST_ASSERT_TRUE(near_value(actual->values[i], tanhf(mat->values[i])));
    }
    sm_destroy(actual);

    actual = sm_clone(mat);
    TEST_ASSERT_TRUE(sm_apply_gelu_tanh(actual));
    for (size_t i = 0; i < n; ++i) {
      TEST_ASSERT_TRUE(
          near_value(actual->values[i], reference_gelu_tanh(mat->values[i])));
    }
    sm_destroy(actual);

    actual = sm_clone(mat);
    TEST_ASSERT_TRUE(sm_apply_gelu_erf(actual));
    for (size_t i = 0; i < n; ++i) {
      TEST_ASSERT_TRUE(
          near_value(actual->values[i], reference_gelu_erf(mat->values[i])));
    }
    sm_destroy(actual);
    sm_destroy(mat);
  }
}

void test_sm_activation_kernels_reject_empty_buffers_and_propagate_nan(void) {
  float value = NAN;
  TEST_ASSERT_FALSE(sm_kernel_apply_sigmoid(NULL, 1));
  TEST_ASSERT_FALSE(sm_kernel_apply_sigmoid(&value, 0));

  FloatMatrix *mat = sm_create(1, 1);
  TEST_ASSERT_NOT_NULL(mat);
  mat->values[0] = NAN;
  TEST_ASSERT_TRUE(sm_apply_sigmoid(mat));
  TEST_ASSERT_TRUE(isnan(mat->values[0]));
  mat->values[0] = NAN;
  TEST_ASSERT_TRUE(sm_apply_tanh(mat));
  TEST_ASSERT_TRUE(isnan(mat->values[0]));
  mat->values[0] = NAN;
  TEST_ASSERT_TRUE(sm_apply_gelu_tanh(mat));
  TEST_ASSERT_TRUE(isnan(mat->values[0]));
  mat->values[0] = NAN;
  TEST_ASSERT_TRUE(sm_apply_gelu_erf(mat));
  TEST_ASSERT_TRUE(isnan(mat->values[0]));
  sm_destroy(mat);

  float nan_value = NAN;
  const float zero_bias = 0.0f;
  uint32_t initial_nan_bits = 0;
  memcpy(&initial_nan_bits, &nan_value, sizeof(initial_nan_bits));
  TEST_ASSERT_EQUAL_UINT32(0x7f800000u,
                           initial_nan_bits & 0x7f800000u);
  TEST_ASSERT_TRUE(sm_kernel_add_bias_activation(
      &nan_value, &zero_bias, 1, 1, 1, SM_ACT_RELU));
  uint32_t nan_bits = 0;
  memcpy(&nan_bits, &nan_value, sizeof(nan_bits));
  TEST_ASSERT_EQUAL_UINT32(0x7f800000u, nan_bits & 0x7f800000u);
  TEST_ASSERT_TRUE((nan_bits & 0x007fffffu) != 0);
}

void test_sm_softmax_is_stable_for_large_values_and_propagates_nan(void) {
  FloatMatrix *mat = sm_create_with_values(
      2, 3, (float[]){1000.0f, 999.0f, -1000.0f, -1000.0f, -1001.0f,
                      -1002.0f});
  TEST_ASSERT_NOT_NULL(mat);
  TEST_ASSERT_TRUE(sm_apply_softmax(mat));
  for (size_t row = 0; row < 2; ++row) {
    float sum = 0.0f;
    for (size_t col = 0; col < 3; ++col) {
      const float v = mat->values[row * 3 + col];
      TEST_ASSERT_TRUE(isfinite(v));
      sum += v;
    }
    TEST_ASSERT_TRUE(near_value(sum, 1.0f));
  }
  sm_destroy(mat);

  mat = sm_create_with_values(1, 3, (float[]){0.0f, NAN, 1.0f});
  TEST_ASSERT_NOT_NULL(mat);
  TEST_ASSERT_TRUE(sm_apply_softmax(mat));
  TEST_ASSERT_TRUE(isnan(mat->values[0]));
  TEST_ASSERT_TRUE(isnan(mat->values[1]));
  TEST_ASSERT_TRUE(isnan(mat->values[2]));
  sm_destroy(mat);
}

void test_sm_backward_kernels_match_numerical_derivatives(void) {
  const float inputs[] = {-1.7f, -0.25f, 0.8f, 2.1f};
  const size_t n = sizeof(inputs) / sizeof(inputs[0]);
  const float original_grads[] = {0.5f, -1.0f, 2.0f, 3.0f};
  const float h = 1.0e-3f;

  FloatMatrix *x = sm_create_with_values(1, n, (float *)inputs);
  FloatMatrix *activation = sm_clone(x);
  FloatMatrix *grad = sm_create_with_values(1, n, (float *)original_grads);
  TEST_ASSERT_NOT_NULL(x);
  TEST_ASSERT_NOT_NULL(activation);
  TEST_ASSERT_NOT_NULL(grad);
  TEST_ASSERT_TRUE(sm_apply_sigmoid(activation));
  TEST_ASSERT_TRUE(sm_apply_sigmoid_backward(activation, grad));
  for (size_t i = 0; i < n; ++i) {
    const float numeric =
        (reference_sigmoid(inputs[i] + h) -
         reference_sigmoid(inputs[i] - h)) /
        (2.0f * h) * original_grads[i];
    TEST_ASSERT_FLOAT_WITHIN(2.0e-4f, numeric, grad->values[i]);
  }
  sm_destroy(activation);
  sm_destroy(grad);

  activation = sm_clone(x);
  grad = sm_create_with_values(1, n, (float *)original_grads);
  TEST_ASSERT_TRUE(sm_apply_tanh(activation));
  TEST_ASSERT_TRUE(sm_apply_tanh_backward(activation, grad));
  for (size_t i = 0; i < n; ++i) {
    const float numeric =
      (tanhf(inputs[i] + h) - tanhf(inputs[i] - h)) /
      (2.0f * h) * original_grads[i];
    TEST_ASSERT_FLOAT_WITHIN(2.0e-4f, numeric, grad->values[i]);
  }
  sm_destroy(activation);
  sm_destroy(grad);

  for (int variant = 0; variant < 2; ++variant) {
    grad = sm_create_with_values(1, n, (float *)original_grads);
    TEST_ASSERT_NOT_NULL(grad);
    bool ok = variant == 0 ? sm_apply_gelu_tanh_backward(x, grad)
                           : sm_apply_gelu_erf_backward(x, grad);
    TEST_ASSERT_TRUE(ok);
    for (size_t i = 0; i < n; ++i) {
      const float xp = inputs[i] + h;
      const float xm = inputs[i] - h;
      const float yp = variant == 0 ? reference_gelu_tanh(xp)
                                    : reference_gelu_erf(xp);
      const float ym = variant == 0 ? reference_gelu_tanh(xm)
                                    : reference_gelu_erf(xm);
      const float numeric = (yp - ym) / (2.0f * h) * original_grads[i];
      TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, numeric, grad->values[i]);
    }
    sm_destroy(grad);
  }
  sm_destroy(x);
}

void test_sm_relu_backward_matches_finite_difference(void) {
  const float inputs[] = {-2.0f, -0.01f, 0.02f, 3.0f};
  const float original_grad[] = {1.0f, -2.0f, 0.5f, 3.0f};
  FloatMatrix *activation = sm_create_with_values(1, 4, (float *)inputs);
  FloatMatrix *grad = sm_create_with_values(1, 4, (float *)original_grad);
  TEST_ASSERT_NOT_NULL(activation);
  TEST_ASSERT_NOT_NULL(grad);
  TEST_ASSERT_TRUE(sm_apply_relu(activation));
  TEST_ASSERT_TRUE(sm_apply_relu_backward(activation, grad));
  const float h = 1.0e-3f;
  for (size_t i = 0; i < 4; ++i) {
    const float plus = fmaxf(0.0f, inputs[i] + h);
    const float minus = fmaxf(0.0f, inputs[i] - h);
    const float numeric = (plus - minus) / (2.0f * h) * original_grad[i];
    TEST_ASSERT_FLOAT_WITHIN(5.0e-4f, numeric, grad->values[i]);
  }
  sm_destroy(grad);
  sm_destroy(activation);
}

void test_sm_elementwise_activation_results_are_thread_count_independent(void) {
  FloatMatrix *input = sm_create(1, 100000);
  FloatMatrix *one_thread = sm_create(1, 100000);
  FloatMatrix *many_threads = sm_create(1, 100000);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(one_thread);
  TEST_ASSERT_NOT_NULL(many_threads);
  for (size_t i = 0; i < input->cols; ++i) {
    input->values[i] = sinf((float)i * 0.003f);
  }
  memcpy(one_thread->values, input->values, input->cols * sizeof(float));
  memcpy(many_threads->values, input->values, input->cols * sizeof(float));

  const int previous_threads = omp_get_max_threads();
  omp_set_dynamic(0);
  for (int activation = SM_ACT_RELU; activation <= SM_ACT_TANH; ++activation) {
    memcpy(one_thread->values, input->values, input->cols * sizeof(float));
    memcpy(many_threads->values, input->values, input->cols * sizeof(float));
    omp_set_num_threads(1);
    TEST_ASSERT_TRUE(sm_kernel_apply_activation(one_thread->values,
                                                one_thread->cols,
                                                (SmActivation)activation));
    omp_set_num_threads(8);
    TEST_ASSERT_TRUE(sm_kernel_apply_activation(many_threads->values,
                                                many_threads->cols,
                                                (SmActivation)activation));
    TEST_ASSERT_EQUAL(0, memcmp(one_thread->values, many_threads->values,
                                input->cols * sizeof(float)));
  }
  omp_set_num_threads(1);
  TEST_ASSERT_TRUE(sm_apply_gelu_tanh(one_thread));
  omp_set_num_threads(8);
  TEST_ASSERT_TRUE(sm_apply_gelu_tanh(many_threads));
  TEST_ASSERT_EQUAL(0, memcmp(one_thread->values, many_threads->values,
                              input->cols * sizeof(float)));
  omp_set_num_threads(previous_threads);

  sm_destroy(input);
  sm_destroy(one_thread);
  sm_destroy(many_threads);
}

void test_sm_gemm_bias_act_matches_reference_with_transposes_and_bias_shapes(void) {
  FloatMatrix *a = sm_create_with_values(
      3, 2, (float[]){1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 2.0f});
  FloatMatrix *b = sm_create_with_values(
      4, 3, (float[]){1.0f, 2.0f, -1.0f, -2.0f, 0.5f, 1.0f,
                      3.0f, -1.0f, 2.0f, 0.25f, 1.5f, -0.5f});
  FloatMatrix *bias = sm_create_with_values(
      1, 4, (float[]){0.25f, -0.5f, 1.0f, -1.0f});
  FloatMatrix *full_bias = sm_create_with_values(
      2, 4, (float[]){0.25f, -0.5f, 1.0f, -1.0f,
                      -0.25f, 0.5f, -1.0f, 1.0f});
  FloatMatrix *out = sm_create(2, 4);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);
  TEST_ASSERT_NOT_NULL(bias);
  TEST_ASSERT_NOT_NULL(full_bias);
  TEST_ASSERT_NOT_NULL(out);

  for (int act = SM_ACT_NONE; act <= SM_ACT_TANH; ++act) {
    const SmActivation activation = (SmActivation)act;
    TEST_ASSERT_TRUE(sm_gemm_bias_act(out, a, SM_TRANSPOSE, b, SM_TRANSPOSE,
                                      bias, activation));
    for (size_t row = 0; row < out->rows; ++row) {
      for (size_t col = 0; col < out->cols; ++col) {
        float sum = 0.0f;
        for (size_t k = 0; k < 3; ++k) {
          sum += a->values[k * a->cols + row] *
                 b->values[col * b->cols + k];
        }
        const float expected = reference_activation(
            activation, sum + bias->values[col]);
        TEST_ASSERT_TRUE(near_value(out->values[row * out->cols + col],
                                    expected));
      }
    }
  }

  TEST_ASSERT_TRUE(sm_gemm_bias_act(out, a, SM_TRANSPOSE, b, SM_TRANSPOSE,
                                    full_bias, SM_ACT_RELU));
  float expected_relu[8];
  memcpy(expected_relu, out->values, sizeof(expected_relu));
  TEST_ASSERT_TRUE(sm_gemm_bias_relu(out, a, SM_TRANSPOSE, b, SM_TRANSPOSE,
                                     full_bias));
  for (size_t i = 0; i < out->rows * out->cols; ++i) {
    TEST_ASSERT_TRUE(out->values[i] >= 0.0f);
    TEST_ASSERT_TRUE(near_value(out->values[i], expected_relu[i]));
  }
  TEST_ASSERT_TRUE(sm_gemm_bias_act(out, a, SM_TRANSPOSE, b, SM_TRANSPOSE,
                                    NULL, SM_ACT_TANH));
  for (size_t row = 0; row < out->rows; ++row) {
    for (size_t col = 0; col < out->cols; ++col) {
      float sum = 0.0f;
      for (size_t k = 0; k < 3; ++k) {
        sum += a->values[k * a->cols + row] * b->values[col * b->cols + k];
      }
      TEST_ASSERT_TRUE(near_value(out->values[row * out->cols + col],
                                  tanhf(sum)));
    }
  }
  TEST_ASSERT_FALSE(sm_gemm_bias_act(out, a, SM_TRANSPOSE, b, SM_TRANSPOSE,
                                     a, SM_ACT_RELU));
  TEST_ASSERT_FALSE(sm_gemm_bias_act(out, a, SM_TRANSPOSE, b, SM_TRANSPOSE,
                                     NULL, (SmActivation)99));

  sm_destroy(out);
  sm_destroy(full_bias);
  sm_destroy(bias);
  sm_destroy(b);
  sm_destroy(a);
}

void test_sm_row_helpers_and_losses_match_nm_lib_semantics(void) {
  FloatMatrix *mat = sm_create_with_values(
      3, 4, (float[]){1.0f, 5.0f, 5.0f, 0.0f,
                      -1.0f, -2.0f, -3.0f, -4.0f,
                      2.0f, 2.0f, 3.0f, 3.0f});
  FloatMatrix *row = sm_create_with_values(
      1, 4, (float[]){10.0f, -1.0f, -1.0f, 2.0f});
  TEST_ASSERT_NOT_NULL(mat);
  TEST_ASSERT_NOT_NULL(row);
  TEST_ASSERT_TRUE(sm_add_rowwise(mat, row));
  const float expected_values[] = {11.0f, 4.0f, 4.0f, 2.0f,
                                   9.0f, -3.0f, -4.0f, -2.0f,
                                   12.0f, 1.0f, 2.0f, 5.0f};
  for (size_t i = 0; i < 12; ++i) {
    TEST_ASSERT_TRUE(near_value(mat->values[i], expected_values[i]));
  }

  FloatMatrix *sum = sm_sum_rows(mat);
  FloatMatrix *argmax = sm_argmax_rows(mat);
  TEST_ASSERT_NOT_NULL(sum);
  TEST_ASSERT_NOT_NULL(argmax);
  const float expected_sum[] = {32.0f, 2.0f, 2.0f, 5.0f};
  for (size_t col = 0; col < 4; ++col) {
    TEST_ASSERT_TRUE(near_value(sum->values[col], expected_sum[col]));
  }
  TEST_ASSERT_EQUAL_FLOAT(1.0f, argmax->values[0]);
  TEST_ASSERT_EQUAL_FLOAT(0.0f, argmax->values[1]);
  TEST_ASSERT_EQUAL_FLOAT(0.0f, argmax->values[2]);

  FloatMatrix *predicted = sm_create_with_values(
      2, 2, (float[]){0.5f, 0.5f, 0.25f, 0.75f});
  FloatMatrix *target = sm_create_with_values(
      2, 2, (float[]){1.0f, 0.0f, 0.0f, 1.0f});
  FloatMatrix *grad = sm_create(2, 2);
  TEST_ASSERT_NOT_NULL(predicted);
  TEST_ASSERT_NOT_NULL(target);
  TEST_ASSERT_NOT_NULL(grad);
  TEST_ASSERT_TRUE(near_value(sm_loss_mse(predicted, target), 0.15625f));
  TEST_ASSERT_TRUE(near_value(
      sm_loss_cross_entropy(predicted, target),
      (-logf(0.5f) - logf(0.75f)) / 2.0f));
  TEST_ASSERT_TRUE(sm_softmax_ce_grad(predicted, target, grad));
  TEST_ASSERT_TRUE(near_value(grad->values[0], -0.5f));
  TEST_ASSERT_TRUE(near_value(grad->values[1], 0.5f));
  TEST_ASSERT_TRUE(near_value(grad->values[2], 0.25f));
  TEST_ASSERT_TRUE(near_value(grad->values[3], -0.25f));

  TEST_ASSERT_FALSE(sm_add_rowwise(mat, predicted));
  TEST_ASSERT_EQUAL_FLOAT(-1.0f, sm_loss_mse(predicted, row));
  TEST_ASSERT_FALSE(sm_softmax_ce_grad(predicted, target, row));

  sm_destroy(grad);
  sm_destroy(target);
  sm_destroy(predicted);
  sm_destroy(argmax);
  sm_destroy(sum);
  sm_destroy(row);
  sm_destroy(mat);
}

void test_sm_rowwise_add_handles_aliased_bias_and_sum_is_thread_deterministic(void) {
  FloatMatrix *aliased = sm_create_with_values(
      2, 3, (float[]){1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f});
  TEST_ASSERT_NOT_NULL(aliased);
  FloatMatrix row_view = {
      .rows = 1,
      .cols = 3,
      .capacity = 3,
      .values = aliased->values,
  };
  TEST_ASSERT_TRUE(sm_add_rowwise(aliased, &row_view));
  const float expected[] = {2.0f, 4.0f, 6.0f, 11.0f, 22.0f, 33.0f};
  for (size_t i = 0; i < 6; ++i) {
    TEST_ASSERT_TRUE(near_value(aliased->values[i], expected[i]));
  }
  sm_destroy(aliased);

  FloatMatrix *many_rows = sm_create(4096, 7);
  TEST_ASSERT_NOT_NULL(many_rows);
  for (size_t i = 0; i < many_rows->rows * many_rows->cols; ++i) {
    many_rows->values[i] = (float)((i * 29u) % 101u) * 0.01f - 0.5f;
  }
  const int previous_threads = omp_get_max_threads();
  omp_set_dynamic(0);
  omp_set_num_threads(1);
  FloatMatrix *one_thread = sm_sum_rows(many_rows);
  omp_set_num_threads(8);
  FloatMatrix *eight_threads = sm_sum_rows(many_rows);
  TEST_ASSERT_NOT_NULL(one_thread);
  TEST_ASSERT_NOT_NULL(eight_threads);
  TEST_ASSERT_EQUAL(0, memcmp(one_thread->values, eight_threads->values,
                              many_rows->cols * sizeof(float)));
  omp_set_num_threads(previous_threads);

  sm_destroy(eight_threads);
  sm_destroy(one_thread);
  sm_destroy(many_rows);
}

void test_sm_row_kernels_and_losses_match_scalar_references_sizes(void) {
  const size_t widths[] = {1, 7, 1000, 100000};
  for (size_t s = 0; s < sizeof(widths) / sizeof(widths[0]); ++s) {
    const size_t rows = 4;
    const size_t cols = widths[s];
    const size_t count = rows * cols;
    FloatMatrix *mat = sm_create(rows, cols);
    FloatMatrix *bias = sm_create(1, cols);
    FloatMatrix *target = sm_create(rows, cols);
    TEST_ASSERT_NOT_NULL(mat);
    TEST_ASSERT_NOT_NULL(bias);
    TEST_ASSERT_NOT_NULL(target);
    for (size_t col = 0; col < cols; ++col) {
      bias->values[col] = 0.25f * cosf((float)col * 0.017f);
    }
    for (size_t i = 0; i < count; ++i) {
      mat->values[i] = sinf((float)i * 0.011f);
      target->values[i] = (float)((i * 13u) % 19u) / 20.0f;
    }

    TEST_ASSERT_TRUE(sm_add_rowwise(mat, bias));
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        const float original = sinf((float)(row * cols + col) * 0.011f);
        TEST_ASSERT_TRUE(near_value(mat->values[row * cols + col],
                                    original + bias->values[col]));
      }
    }

    FloatMatrix *sum = sm_sum_rows(mat);
    FloatMatrix *argmax = sm_argmax_rows(mat);
    TEST_ASSERT_NOT_NULL(sum);
    TEST_ASSERT_NOT_NULL(argmax);
    for (size_t col = 0; col < cols; ++col) {
      float expected_sum = 0.0f;
      for (size_t row = 0; row < rows; ++row) {
        expected_sum += mat->values[row * cols + col];
      }
      TEST_ASSERT_EQUAL_FLOAT(expected_sum, sum->values[col]);
    }
    for (size_t row = 0; row < rows; ++row) {
      size_t best = 0;
      for (size_t col = 1; col < cols; ++col) {
        if (mat->values[row * cols + col] > mat->values[row * cols + best]) {
          best = col;
        }
      }
      TEST_ASSERT_EQUAL_FLOAT((float)best, argmax->values[row]);
    }

    float expected_mse = 0.0f;
    float expected_ce = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      const float difference = mat->values[i] - target->values[i];
      expected_mse += difference * difference;
      const float probability = fmaxf(mat->values[i], 1.0e-7f);
      expected_ce += -target->values[i] * logf(probability);
    }
    expected_mse /= (float)count;
    expected_ce /= (float)rows;
    TEST_ASSERT_EQUAL_FLOAT(expected_mse, sm_loss_mse(mat, target));
    TEST_ASSERT_EQUAL_FLOAT(expected_ce, sm_loss_cross_entropy(mat, target));

    sm_destroy(argmax);
    sm_destroy(sum);
    sm_destroy(target);
    sm_destroy(bias);
    sm_destroy(mat);
  }

  FloatMatrix empty = {0};
  TEST_ASSERT_FALSE(sm_add_rowwise(&empty, &empty));
  TEST_ASSERT_NULL(sm_sum_rows(&empty));
  TEST_ASSERT_NULL(sm_argmax_rows(&empty));
  TEST_ASSERT_EQUAL_FLOAT(-1.0f, sm_loss_mse(&empty, &empty));
  TEST_ASSERT_EQUAL_FLOAT(-1.0f, sm_loss_cross_entropy(&empty, &empty));
}