/*
 * SPDX-License-Identifier: MIT
 */

#include "dv.h"
#include "dvs.h"

#include <math.h>
#include <stdint.h>

#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_dvs_set_get_and_compact_should_work(void) {
  DoubleSparseVector *vec = dvs_create(8, 2);
  TEST_ASSERT_NOT_NULL(vec);

  TEST_ASSERT_TRUE(dvs_set(vec, 3, 2.5));
  TEST_ASSERT_TRUE(dvs_set(vec, 1, 1.5));
  TEST_ASSERT_TRUE(dvs_set(vec, 3, 4.5));
  TEST_ASSERT_TRUE(fabs(dvs_get(vec, 1) - 1.5) < 1e-10);
  TEST_ASSERT_TRUE(fabs(dvs_get(vec, 3) - 4.5) < 1e-10);
  TEST_ASSERT_TRUE(dvs_sort_indices(vec));
  TEST_ASSERT_TRUE(dvs_compact(vec));
  TEST_ASSERT_EQUAL_UINT32(2, (uint32_t)vec->nnz);

  dvs_destroy(vec);
}

void test_dvs_add_and_dot_dense_should_work(void) {
  DoubleSparseVector *lhs = dvs_create(4, 2);
  DoubleSparseVector *rhs = dvs_create(4, 2);
  TEST_ASSERT_TRUE(dvs_set(lhs, 0, 1.0));
  TEST_ASSERT_TRUE(dvs_set(lhs, 2, 3.0));
  TEST_ASSERT_TRUE(dvs_set(rhs, 2, 4.0));
  TEST_ASSERT_TRUE(dvs_set(rhs, 3, 5.0));

  DoubleSparseVector *sum = dvs_add(lhs, rhs);
  /* sum = {0:1, 2:7, 3:5}, dense = {2, 0, 10, 1} => dot = 1*2 + 7*10 + 5*1 = 77 */
  const double dense_values[] = {2.0, 0.0, 10.0, 1.0};
  DoubleVector *dense = dv_create_with_values(4, dense_values);

  TEST_ASSERT_NOT_NULL(sum);
  TEST_ASSERT_TRUE(fabs(dvs_get(sum, 0) - 1.0) < 1e-10);
  TEST_ASSERT_TRUE(fabs(dvs_get(sum, 2) - 7.0) < 1e-10);
  TEST_ASSERT_TRUE(fabs(dvs_get(sum, 3) - 5.0) < 1e-10);
  TEST_ASSERT_TRUE(fabs(dvs_dot_dense(sum, dense) - 77.0) < 1e-10);

  dv_destroy(dense);
  dvs_destroy(sum);
  dvs_destroy(rhs);
  dvs_destroy(lhs);
}
