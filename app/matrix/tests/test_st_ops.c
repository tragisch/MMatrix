/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st.h"

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
#ifndef TEST_ASSERT_NULL
#define TEST_ASSERT_NULL(value) \
  do {                          \
    (void)(value);              \
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

#define EPSILON 1e-5f

void setUp(void) {}
void tearDown(void) {}

/* ---- st_inplace_add ---- */

void test_st_inplace_add_should_add_elementwise(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *a = st_create(2, shape);
  FloatTensor *b = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);

  for (size_t i = 0; i < 6; ++i) {
    a->values[i] = (float)i;
    b->values[i] = 10.0f;
  }

  bool ok = st_inplace_add(a, b);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 6; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, (float)i + 10.0f, a->values[i]);
  }

  st_destroy(b);
  st_destroy(a);
}

void test_st_inplace_add_null_b_should_be_noop(void) {
  size_t shape[1] = {3};
  FloatTensor *a = st_create(1, shape);
  TEST_ASSERT_NOT_NULL(a);
  a->values[0] = 1.0f;

  bool ok = st_inplace_add(a, NULL);
  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, a->values[0]);

  st_destroy(a);
}

/* ---- st_inplace_sub ---- */

void test_st_inplace_sub_should_subtract_elementwise(void) {
  size_t shape[1] = {4};
  FloatTensor *a = st_create(1, shape);
  FloatTensor *b = st_create(1, shape);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);

  for (size_t i = 0; i < 4; ++i) {
    a->values[i] = 10.0f;
    b->values[i] = (float)i;
  }

  bool ok = st_inplace_sub(a, b);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 4; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 10.0f - (float)i, a->values[i]);
  }

  st_destroy(b);
  st_destroy(a);
}

/* ---- st_inplace_scale ---- */

void test_st_inplace_scale_should_multiply_all(void) {
  size_t shape[1] = {3};
  FloatTensor *t = st_create(1, shape);
  TEST_ASSERT_NOT_NULL(t);

  t->values[0] = 1.0f;
  t->values[1] = 2.0f;
  t->values[2] = 3.0f;

  bool ok = st_inplace_scale(t, 0.5f);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.5f, t->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, t->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.5f, t->values[2]);

  st_destroy(t);
}

/* ---- st_inplace_elementwise_multiply ---- */

void test_st_inplace_elementwise_multiply_should_hadamard(void) {
  size_t shape[1] = {3};
  FloatTensor *a = st_create(1, shape);
  FloatTensor *b = st_create(1, shape);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);

  float va[] = {2.0f, 3.0f, 4.0f};
  float vb[] = {5.0f, 6.0f, 7.0f};
  memcpy(a->values, va, sizeof(va));
  memcpy(b->values, vb, sizeof(vb));

  bool ok = st_inplace_elementwise_multiply(a, b);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 10.0f, a->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 18.0f, a->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 28.0f, a->values[2]);

  st_destroy(b);
  st_destroy(a);
}

/* ---- st_fill ---- */

void test_st_fill_should_set_all_elements(void) {
  size_t shape[2] = {3, 4};
  FloatTensor *t = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(t);

  bool ok = st_fill(t, 42.0f);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 12; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 42.0f, t->values[i]);
  }

  st_destroy(t);
}

void test_st_fill_zero_should_zero(void) {
  size_t shape[1] = {5};
  FloatTensor *t = st_create(1, shape);
  TEST_ASSERT_NOT_NULL(t);
  for (size_t i = 0; i < 5; ++i) {
    t->values[i] = 99.0f;
  }

  bool ok = st_fill(t, 0.0f);
  TEST_ASSERT_TRUE(ok);

  for (size_t i = 0; i < 5; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->values[i]);
  }

  st_destroy(t);
}

/* ---- st_apply_relu ---- */

void test_st_apply_relu_should_clamp_negatives(void) {
  size_t shape[1] = {5};
  FloatTensor *t = st_create(1, shape);
  TEST_ASSERT_NOT_NULL(t);

  float vals[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
  memcpy(t->values, vals, sizeof(vals));

  bool ok = st_apply_relu(t);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->values[2]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, t->values[3]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, t->values[4]);

  st_destroy(t);
}

/* ---- st_apply_relu_backward ---- */

void test_st_apply_relu_backward_should_zero_where_inactive(void) {
  size_t shape[1] = {5};
  FloatTensor *activation = st_create(1, shape);
  FloatTensor *grad = st_create(1, shape);
  TEST_ASSERT_NOT_NULL(activation);
  TEST_ASSERT_NOT_NULL(grad);

  float act_vals[] = {0.0f, 0.0f, 0.0f, 2.0f, 5.0f};
  float grad_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  memcpy(activation->values, act_vals, sizeof(act_vals));
  memcpy(grad->values, grad_vals, sizeof(grad_vals));

  bool ok = st_apply_relu_backward(activation, grad);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, grad->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, grad->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, grad->values[2]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, grad->values[3]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, grad->values[4]);

  st_destroy(grad);
  st_destroy(activation);
}

/* ---- st_sum_axes ---- */

void test_st_sum_axes_should_sum_over_single_axis(void) {
  /* 2x3 tensor, sum over axis 0 → shape [3] */
  size_t shape[2] = {2, 3};
  FloatTensor *t = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(t);

  /* [[1,2,3],[4,5,6]] */
  float vals[] = {1, 2, 3, 4, 5, 6};
  memcpy(t->values, vals, sizeof(vals));

  size_t axes[1] = {0};
  FloatTensor *r = st_sum_axes(t, axes, 1);
  TEST_ASSERT_NOT_NULL(r);

  TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)r->ndim);
  TEST_ASSERT_EQUAL_UINT32(3u, (uint32_t)r->shape[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, r->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 7.0f, r->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 9.0f, r->values[2]);

  st_destroy(r);
  st_destroy(t);
}

void test_st_sum_axes_should_sum_over_multiple_axes(void) {
  /* [2, 3, 4] tensor, sum over axes {0, 2} → shape [3] */
  size_t shape[3] = {2, 3, 4};
  FloatTensor *t = st_create(3, shape);
  TEST_ASSERT_NOT_NULL(t);

  /* Fill with 1.0 */
  for (size_t i = 0; i < 24; ++i) {
    t->values[i] = 1.0f;
  }

  size_t axes[2] = {0, 2};
  FloatTensor *r = st_sum_axes(t, axes, 2);
  TEST_ASSERT_NOT_NULL(r);

  TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)r->ndim);
  TEST_ASSERT_EQUAL_UINT32(3u, (uint32_t)r->shape[0]);
  /* Each: 2 * 4 = 8 */
  for (size_t i = 0; i < 3; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 8.0f, r->values[i]);
  }

  st_destroy(r);
  st_destroy(t);
}

void test_st_sum_axes_nchw_bias_gradient(void) {
  /* Sum over {0, 2, 3} of NCHW → [C] (bias gradient pattern). */
  size_t shape[4] = {2, 3, 2, 2};
  FloatTensor *t = st_create(4, shape);
  TEST_ASSERT_NOT_NULL(t);

  /* Fill channel ci with value (ci+1). */
  for (size_t ni = 0; ni < 2; ++ni) {
    for (size_t ci = 0; ci < 3; ++ci) {
      for (size_t i = 0; i < 4; ++i) {
        t->values[((ni * 3 + ci) * 4) + i] = (float)(ci + 1);
      }
    }
  }

  size_t axes[3] = {0, 2, 3};
  FloatTensor *r = st_sum_axes(t, axes, 3);
  TEST_ASSERT_NOT_NULL(r);

  TEST_ASSERT_EQUAL_UINT32(1u, (uint32_t)r->ndim);
  TEST_ASSERT_EQUAL_UINT32(3u, (uint32_t)r->shape[0]);
  /* channel 0: 2*4*1=8, channel 1: 2*4*2=16, channel 2: 2*4*3=24. */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 8.0f, r->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 16.0f, r->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 24.0f, r->values[2]);

  st_destroy(r);
  st_destroy(t);
}

/* ---- st_pad_nchw ---- */

void test_st_pad_nchw_should_add_zeros(void) {
  size_t shape[4] = {1, 1, 2, 2};
  FloatTensor *t = st_create(4, shape);
  TEST_ASSERT_NOT_NULL(t);

  float vals[] = {1, 2, 3, 4};
  memcpy(t->values, vals, sizeof(vals));

  FloatTensor *p = st_pad_nchw(t, 1, 1, 0.0f);
  TEST_ASSERT_NOT_NULL(p);

  TEST_ASSERT_EQUAL_UINT32(4u, (uint32_t)p->shape[2]);
  TEST_ASSERT_EQUAL_UINT32(4u, (uint32_t)p->shape[3]);

  /* Top-left corner should be 0. */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, p->values[0]);
  /* (1,1) should be 1.0 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, p->values[5]);
  /* (1,2) should be 2.0 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, p->values[6]);
  /* (2,1) should be 3.0 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, p->values[9]);
  /* (2,2) should be 4.0 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, p->values[10]);
  /* Bottom-right should be 0. */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, p->values[15]);

  st_destroy(p);
  st_destroy(t);
}

void test_st_pad_nchw_no_padding_should_clone(void) {
  size_t shape[4] = {1, 1, 2, 2};
  FloatTensor *t = st_create(4, shape);
  TEST_ASSERT_NOT_NULL(t);
  for (size_t i = 0; i < 4; ++i) {
    t->values[i] = (float)(i + 1);
  }

  FloatTensor *p = st_pad_nchw(t, 0, 0, 0.0f);
  TEST_ASSERT_NOT_NULL(p);
  TEST_ASSERT_TRUE(p->values != t->values); /* deep copy */

  for (size_t i = 0; i < 4; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, t->values[i], p->values[i]);
  }

  st_destroy(p);
  st_destroy(t);
}


