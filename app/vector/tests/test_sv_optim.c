#include "sv_optim.h"

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
  return fabsf(actual - expected) <= 1.0e-6f * fmaxf(1.0f, fabsf(expected));
}

void setUp(void) {}
void tearDown(void) {}

void test_sv_optimizers_match_scalar_reference_sizes(void) {
  const size_t sizes[] = {1, 7, 1000, 100000};
  const float learning_rate = 0.025f;
  const float grad_scale = 0.4f;
  const float weight_decay = 0.003f;
  const float beta1 = 0.9f;
  const float beta2 = 0.99f;
  const float epsilon = 1.0e-7f;
  const uint64_t step = 7;

  for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
    const size_t n = sizes[si];
    float *params = (float *)malloc(n * sizeof(float));
    float *grads = (float *)malloc(n * sizeof(float));
    float *state1 = (float *)malloc(n * sizeof(float));
    float *state2 = (float *)malloc(n * sizeof(float));
    TEST_ASSERT_NOT_NULL(params);
    TEST_ASSERT_NOT_NULL(grads);
    TEST_ASSERT_NOT_NULL(state1);
    TEST_ASSERT_NOT_NULL(state2);

    for (size_t i = 0; i < n; ++i) {
      params[i] = 0.25f + (float)(i % 23u) * 0.01f;
      grads[i] = ((float)(i % 19u) - 9.0f) * 0.003f;
      state1[i] = 0.02f;
      state2[i] = 0.03f;
    }

    for (size_t i = 0; i < n; ++i) {
      const float gradient = grads[i] * grad_scale;
      const float expected = params[i] - learning_rate * weight_decay * params[i] -
                             learning_rate * gradient;
      params[i] = 0.25f + (float)(i % 23u) * 0.01f;
      TEST_ASSERT_TRUE(sv_optim_sgd(params + i, grads + i, 1, learning_rate,
                                    grad_scale, weight_decay));
      TEST_ASSERT_TRUE(near_value(params[i], expected));
    }

    for (size_t i = 0; i < n; ++i) {
      params[i] = 0.25f + (float)(i % 23u) * 0.01f;
      state1[i] = 0.02f;
    }
    TEST_ASSERT_TRUE(sv_optim_momentum(params, grads, state1, n,
                                      learning_rate, 0.8f, grad_scale,
                                      weight_decay));
    for (size_t i = 0; i < n; ++i) {
      const float gradient = grads[i] * grad_scale;
      const float velocity = 0.8f * 0.02f + gradient;
      const float original = 0.25f + (float)(i % 23u) * 0.01f;
      const float expected = original - learning_rate * weight_decay * original -
                             learning_rate * velocity;
      TEST_ASSERT_TRUE(near_value(state1[i], velocity));
      TEST_ASSERT_TRUE(near_value(params[i], expected));
    }

    for (size_t i = 0; i < n; ++i) {
      params[i] = 0.25f + (float)(i % 23u) * 0.01f;
      state2[i] = 0.03f;
    }
    TEST_ASSERT_TRUE(sv_optim_rmsprop(params, grads, state2, n, learning_rate,
                                     beta2, epsilon, grad_scale, weight_decay));
    for (size_t i = 0; i < n; ++i) {
      const float gradient = grads[i] * grad_scale;
      const float square = beta2 * 0.03f + (1.0f - beta2) * gradient * gradient;
      const float original = 0.25f + (float)(i % 23u) * 0.01f;
      const float expected = original - learning_rate * weight_decay * original -
                             learning_rate * gradient / (sqrtf(square) + epsilon);
      TEST_ASSERT_TRUE(near_value(state2[i], square));
      TEST_ASSERT_TRUE(near_value(params[i], expected));
    }

    for (size_t i = 0; i < n; ++i) {
      params[i] = 0.25f + (float)(i % 23u) * 0.01f;
      state1[i] = 0.02f;
      state2[i] = 0.03f;
    }
    TEST_ASSERT_TRUE(sv_optim_adam(params, grads, state1, state2, n,
                                   learning_rate, beta1, beta2, epsilon, step,
                                   grad_scale, weight_decay));
    const float inverse_bias1 = 1.0f / (1.0f - powf(beta1, (float)step));
    const float inverse_bias2 = 1.0f / (1.0f - powf(beta2, (float)step));
    for (size_t i = 0; i < n; ++i) {
      const float gradient = grads[i] * grad_scale;
      const float first = beta1 * 0.02f + (1.0f - beta1) * gradient;
      const float second = beta2 * 0.03f + (1.0f - beta2) * gradient * gradient;
      const float original = 0.25f + (float)(i % 23u) * 0.01f;
      const float expected = original - learning_rate * weight_decay * original -
                             learning_rate * first * inverse_bias1 /
                                 (sqrtf(second * inverse_bias2) + epsilon);
      TEST_ASSERT_TRUE(near_value(state1[i], first));
      TEST_ASSERT_TRUE(near_value(state2[i], second));
      TEST_ASSERT_TRUE(near_value(params[i], expected));
    }

    free(state2);
    free(state1);
    free(grads);
    free(params);
  }
}

void test_sv_optimizers_validate_inputs_and_empty_lengths(void) {
  float value = 1.0f;
  TEST_ASSERT_FALSE(sv_optim_sgd(NULL, &value, 1, 0.1f, 1.0f, 0.0f));
  TEST_ASSERT_FALSE(sv_optim_sgd(&value, &value, 0, 0.1f, 1.0f, 0.0f));
  TEST_ASSERT_FALSE(sv_optim_momentum(&value, &value, NULL, 1, 0.1f, 0.9f,
                                      1.0f, 0.0f));
  TEST_ASSERT_FALSE(sv_optim_rmsprop(&value, &value, &value, 1, 0.1f, 1.0f,
                                     1.0e-8f, 1.0f, 0.0f));
  TEST_ASSERT_FALSE(sv_optim_adam(&value, &value, &value, &value, 1, 0.1f,
                                  0.9f, 0.999f, 1.0e-8f, 0, 1.0f, 0.0f));
  TEST_ASSERT_EQUAL_FLOAT(0.0f, sv_clip_norm(NULL, 0, 1.0f));
  TEST_ASSERT_FALSE(sv_optim_sgd(&value, &value, 1, NAN, 1.0f, 0.0f));
  TEST_ASSERT_EQUAL_FLOAT(-1.0f, sv_clip_norm(&value, 1, NAN));
}

void test_sv_clip_norm_returns_old_norm_and_scales_only_when_needed(void) {
  float gradients[] = {6.0f, 8.0f};
  TEST_ASSERT_TRUE(near_value(sv_clip_norm(gradients, 2, 5.0f), 10.0f));
  TEST_ASSERT_TRUE(near_value(gradients[0], 3.0f));
  TEST_ASSERT_TRUE(near_value(gradients[1], 4.0f));

  float below_limit[] = {3.0f, 4.0f};
  TEST_ASSERT_TRUE(near_value(sv_clip_norm(below_limit, 2, 5.0f), 5.0f));
  TEST_ASSERT_EQUAL_FLOAT(3.0f, below_limit[0]);
  TEST_ASSERT_EQUAL_FLOAT(4.0f, below_limit[1]);
}

void test_sv_adam_is_bitwise_independent_of_openmp_thread_count(void) {
  const size_t n = 100000;
  float *params1 = (float *)malloc(n * sizeof(float));
  float *params8 = (float *)malloc(n * sizeof(float));
  float *grads = (float *)malloc(n * sizeof(float));
  float *m1 = (float *)calloc(n, sizeof(float));
  float *m8 = (float *)calloc(n, sizeof(float));
  float *v1 = (float *)calloc(n, sizeof(float));
  float *v8 = (float *)calloc(n, sizeof(float));
  TEST_ASSERT_NOT_NULL(params1);
  TEST_ASSERT_NOT_NULL(params8);
  TEST_ASSERT_NOT_NULL(grads);
  TEST_ASSERT_NOT_NULL(m1);
  TEST_ASSERT_NOT_NULL(m8);
  TEST_ASSERT_NOT_NULL(v1);
  TEST_ASSERT_NOT_NULL(v8);
  for (size_t i = 0; i < n; ++i) {
    params1[i] = params8[i] = 0.1f + (float)(i % 17u) * 0.01f;
    grads[i] = (float)((i * 7u) % 31u) * 0.001f;
  }

  const int previous_threads = omp_get_max_threads();
  omp_set_dynamic(0);
  omp_set_num_threads(1);
  TEST_ASSERT_TRUE(sv_optim_adam(params1, grads, m1, v1, n, 0.001f, 0.9f,
                                 0.999f, 1.0e-8f, 3, 0.75f, 0.0f));
  omp_set_num_threads(8);
  TEST_ASSERT_TRUE(sv_optim_adam(params8, grads, m8, v8, n, 0.001f, 0.9f,
                                 0.999f, 1.0e-8f, 3, 0.75f, 0.0f));
  TEST_ASSERT_EQUAL(0, memcmp(params1, params8, n * sizeof(float)));
  TEST_ASSERT_EQUAL(0, memcmp(m1, m8, n * sizeof(float)));
  TEST_ASSERT_EQUAL(0, memcmp(v1, v8, n * sizeof(float)));
  omp_set_num_threads(previous_threads);

  free(v8);
  free(v1);
  free(m8);
  free(m1);
  free(grads);
  free(params8);
  free(params1);
}