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

void setUp(void) {}
void tearDown(void) {}

void test_sm_activation_kernels_match_scalar_reference_sizes(void) {
  const size_t sizes[] = {1, 7, 1000, 100000};
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
    const size_t n = sizes[s];
    FloatMatrix *mat = sm_create(1, n);
    TEST_ASSERT_NOT_NULL(mat);
    for (size_t i = 0; i < n; ++i) {
      mat->values[i] = 4.0f * sinf((float)i * 0.013f);
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
  const float h = 1.0e-3f;

  FloatMatrix *x = sm_create_with_values(1, n, (float *)inputs);
  FloatMatrix *activation = sm_clone(x);
  FloatMatrix *grad = sm_create_with_values(1, n,
                                             (float[]){0.5f, -1.0f, 2.0f, 3.0f});
  TEST_ASSERT_NOT_NULL(x);
  TEST_ASSERT_NOT_NULL(activation);
  TEST_ASSERT_NOT_NULL(grad);
  TEST_ASSERT_TRUE(sm_apply_sigmoid(activation));
  TEST_ASSERT_TRUE(sm_apply_sigmoid_backward(activation, grad));
  for (size_t i = 0; i < n; ++i) {
    const float y = 1.0f / (1.0f + expf(-inputs[i]));
    const float expected = 1.0f * y * (1.0f - y) *
                           (i == 0 ? 0.5f : i == 1 ? -1.0f : i == 2 ? 2.0f : 3.0f);
    TEST_ASSERT_TRUE(near_value(grad->values[i], expected));
  }
  sm_destroy(activation);
  sm_destroy(grad);

  activation = sm_clone(x);
  grad = sm_create_with_values(1, n,
                               (float[]){0.5f, -1.0f, 2.0f, 3.0f});
  TEST_ASSERT_TRUE(sm_apply_tanh(activation));
  TEST_ASSERT_TRUE(sm_apply_tanh_backward(activation, grad));
  for (size_t i = 0; i < n; ++i) {
    const float y = tanhf(inputs[i]);
    const float original_grad =
        i == 0 ? 0.5f : i == 1 ? -1.0f : i == 2 ? 2.0f : 3.0f;
    TEST_ASSERT_TRUE(
        near_value(grad->values[i], original_grad * (1.0f - y * y)));
  }
  sm_destroy(activation);
  sm_destroy(grad);

  const float original_grads[] = {0.5f, -1.0f, 2.0f, 3.0f};
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