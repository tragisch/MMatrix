#include "st.h"

#include <math.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 6
#define TEST_CASE(...)
#include "unity.h"

static bool near_value(float actual, float expected) {
  return fabsf(actual - expected) <= 1.0e-6f + 1.0e-5f * fabsf(expected);
}

static bool near_bf16(float actual, float expected) {
  return fabsf(actual - expected) <= 1.0e-2f + 1.0e-2f * fabsf(expected);
}

void setUp(void) {}
void tearDown(void) {}

void test_st_activation_kernels_match_matrix_and_scalar_reference(void) {
  const size_t shape[2] = {2, 7};
  FloatTensor *tensor = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(tensor);
  for (size_t i = 0; i < tensor->numel; ++i) {
    tensor->values[i] = ((float)i - 6.0f) * 0.35f;
  }

  FloatTensor *actual = st_clone(tensor);
  TEST_ASSERT_TRUE(st_apply_sigmoid(actual));
  for (size_t i = 0; i < tensor->numel; ++i) {
    TEST_ASSERT_TRUE(near_value(actual->values[i],
                                1.0f / (1.0f + expf(-tensor->values[i]))));
  }
  st_destroy(actual);

  actual = st_clone(tensor);
  TEST_ASSERT_TRUE(st_apply_tanh(actual));
  for (size_t i = 0; i < tensor->numel; ++i) {
    TEST_ASSERT_TRUE(near_value(actual->values[i], tanhf(tensor->values[i])));
  }
  st_destroy(actual);

  actual = st_clone(tensor);
  TEST_ASSERT_TRUE(st_apply_gelu_tanh(actual));
  for (size_t i = 0; i < tensor->numel; ++i) {
    const float x = tensor->values[i];
    const float expected =
        0.5f * x * (1.0f + tanhf(0.7978845608f *
                                 (x + 0.044715f * x * x * x)));
    TEST_ASSERT_TRUE(near_value(actual->values[i], expected));
  }
  st_destroy(actual);

  TEST_ASSERT_TRUE(st_apply_softmax(tensor));
  for (size_t row = 0; row < shape[0]; ++row) {
    float sum = 0.0f;
    for (size_t col = 0; col < shape[1]; ++col) {
      sum += tensor->values[row * shape[1] + col];
    }
    TEST_ASSERT_TRUE(near_value(sum, 1.0f));
  }
  st_destroy(tensor);
}

void test_st_activation_kernels_promote_bf16_and_reject_views(void) {
  const size_t shape[2] = {1, 7};
  FloatTensor *f32 = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(f32);
  for (size_t i = 0; i < f32->numel; ++i) {
    f32->values[i] = ((float)i - 3.0f) * 0.5f;
  }
  FloatTensor *bf16 = st_to_bf16(f32);
  TEST_ASSERT_NOT_NULL(bf16);
  FloatTensor *quantized_input = st_to_f32(bf16);
  TEST_ASSERT_NOT_NULL(quantized_input);
  TEST_ASSERT_TRUE(st_apply_gelu_erf(bf16));
  FloatTensor *result = st_to_f32(bf16);
  TEST_ASSERT_NOT_NULL(result);
  for (size_t i = 0; i < result->numel; ++i) {
    const float x = quantized_input->values[i];
    TEST_ASSERT_TRUE(near_bf16(
      result->values[i], 0.5f * x * (1.0f + erff(x * 0.7071067812f))));
  }

  const size_t view_shape[2] = {2, 2};
  const ptrdiff_t view_strides[2] = {3, 1};
  FloatTensor *base = st_create(2, (size_t[]){2, 3});
  TEST_ASSERT_NOT_NULL(base);
  FloatTensor *view = st_view(base, 2, view_shape, view_strides, 0);
  TEST_ASSERT_NOT_NULL(view);
  TEST_ASSERT_FALSE(st_apply_sigmoid(view));

  st_destroy(view);
  st_destroy(base);
  st_destroy(result);
  st_destroy(quantized_input);
  st_destroy(bf16);
  st_destroy(f32);
}

void test_st_activation_backward_promotes_bf16_and_checks_shapes(void) {
  const size_t shape[1] = {7};
  FloatTensor *activation_f32 = st_create(1, shape);
  FloatTensor *grad_f32 = st_create(1, shape);
  TEST_ASSERT_NOT_NULL(activation_f32);
  TEST_ASSERT_NOT_NULL(grad_f32);
  for (size_t i = 0; i < shape[0]; ++i) {
    activation_f32->values[i] = -1.0f + (float)i * 0.3f;
    grad_f32->values[i] = 1.0f + (float)i;
  }
  FloatTensor *activation = st_to_bf16(activation_f32);
  FloatTensor *grad = st_to_bf16(grad_f32);
  TEST_ASSERT_NOT_NULL(activation);
  TEST_ASSERT_NOT_NULL(grad);
  FloatTensor *activation_before = st_to_f32(activation);
  FloatTensor *grad_before = st_to_f32(grad);
  TEST_ASSERT_NOT_NULL(activation_before);
  TEST_ASSERT_NOT_NULL(grad_before);
  TEST_ASSERT_TRUE(st_apply_tanh_backward(activation, grad));
  FloatTensor *activation_back = st_to_f32(activation);
  FloatTensor *grad_back = st_to_f32(grad);
  TEST_ASSERT_NOT_NULL(activation_back);
  TEST_ASSERT_NOT_NULL(grad_back);
  for (size_t i = 0; i < shape[0]; ++i) {
    const float value = activation_before->values[i];
    const float expected = grad_before->values[i] * (1.0f - value * value);
    TEST_ASSERT_FLOAT_WITHIN(0.02f, expected, grad_back->values[i]);
  }

  const size_t wrong_shape[1] = {6};
  FloatTensor *wrong = st_create(1, wrong_shape);
  TEST_ASSERT_NOT_NULL(wrong);
  TEST_ASSERT_FALSE(st_apply_sigmoid_backward(activation, wrong));

  st_destroy(wrong);
  st_destroy(grad_back);
  st_destroy(activation_back);
  st_destroy(grad_before);
  st_destroy(activation_before);
  st_destroy(grad);
  st_destroy(activation);
  st_destroy(grad_f32);
  st_destroy(activation_f32);
}