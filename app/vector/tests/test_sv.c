/*
 * SPDX-License-Identifier: MIT
 */

#include "sv.h"

#include <stdint.h>

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 5
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_sv_create_should_zero_initialize(void) {
  FloatVector *vec = sv_create(4);
  TEST_ASSERT_NOT_NULL(vec);
  TEST_ASSERT_EQUAL_UINT32(4, (uint32_t)vec->len);
  for (size_t i = 0; i < vec->len; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, vec->values[i]);
  }
  sv_destroy(vec);
}

void test_sv_add_dot_and_axpy_should_work(void) {
  const float lhs_values[] = {1.0f, 2.0f, 3.0f};
  const float rhs_values[] = {4.0f, 5.0f, 6.0f};
  FloatVector *lhs = sv_create_with_values(3, lhs_values);
  FloatVector *rhs = sv_create_with_values(3, rhs_values);
  FloatVector *sum = sv_add(lhs, rhs);

  TEST_ASSERT_NOT_NULL(sum);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 5.0f, sum->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 7.0f, sum->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 9.0f, sum->values[2]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 32.0f, sv_dot(lhs, rhs));

  TEST_ASSERT_TRUE(sv_axpy(lhs, 2.0f, rhs));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 9.0f, lhs->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 12.0f, lhs->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 15.0f, lhs->values[2]);

  sv_destroy(sum);
  sv_destroy(rhs);
  sv_destroy(lhs);
}

void test_sv_normalize_and_softmax_should_work(void) {
  const float norm_values[] = {3.0f, 4.0f};
  FloatVector *norm_vec = sv_create_with_values(2, norm_values);
  TEST_ASSERT_TRUE(sv_normalize(norm_vec));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.6f, norm_vec->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.8f, norm_vec->values[1]);
  sv_destroy(norm_vec);

  const float softmax_values[] = {1.0f, 2.0f, 3.0f};
  FloatVector *softmax_vec = sv_create_with_values(3, softmax_values);
  TEST_ASSERT_TRUE(sv_softmax(softmax_vec));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, sv_sum(softmax_vec));
  TEST_ASSERT_EQUAL_UINT32(2, (uint32_t)sv_argmax(softmax_vec));
  sv_destroy(softmax_vec);
}
