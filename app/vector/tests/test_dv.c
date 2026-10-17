/*
 * SPDX-License-Identifier: MIT
 */

#include "dv.h"

#include <math.h>

#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_dv_dot_and_scale_should_work(void) {
  const double lhs_values[] = {1.0, 2.0, 3.0};
  const double rhs_values[] = {4.0, 5.0, 6.0};
  DoubleVector *lhs = dv_create_with_values(3, lhs_values);
  DoubleVector *rhs = dv_create_with_values(3, rhs_values);
  DoubleVector *scaled = dv_scale(lhs, 0.5);

  TEST_ASSERT_NOT_NULL(lhs);
  TEST_ASSERT_NOT_NULL(rhs);
  TEST_ASSERT_NOT_NULL(scaled);
  TEST_ASSERT_TRUE(fabs(dv_dot(lhs, rhs) - 32.0) < 1e-10);
  TEST_ASSERT_TRUE(fabs(scaled->values[0] - 0.5) < 1e-10);
  TEST_ASSERT_TRUE(fabs(scaled->values[1] - 1.0) < 1e-10);
  TEST_ASSERT_TRUE(fabs(scaled->values[2] - 1.5) < 1e-10);

  dv_destroy(scaled);
  dv_destroy(rhs);
  dv_destroy(lhs);
}

void test_dv_axpy_and_normalize_should_work(void) {
  const double src_values[] = {1.0, 0.0, 0.0};
  const double dst_values[] = {0.0, 2.0, 0.0};
  DoubleVector *src = dv_create_with_values(3, src_values);
  DoubleVector *dst = dv_create_with_values(3, dst_values);

  TEST_ASSERT_TRUE(dv_axpy(dst, 3.0, src));
  TEST_ASSERT_TRUE(fabs(dst->values[0] - 3.0) < 1e-10);
  TEST_ASSERT_TRUE(fabs(dst->values[1] - 2.0) < 1e-10);
  TEST_ASSERT_TRUE(dv_normalize(dst));
  TEST_ASSERT_TRUE(fabs(dv_norm_l2(dst) - 1.0) < 1e-10);

  dv_destroy(dst);
  dv_destroy(src);
}
