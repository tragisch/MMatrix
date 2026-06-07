/*
 * SPDX-License-Identifier: MIT
 */

#include "st_vector_bridge.h"

#include <stdint.h>

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 5
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_st_vector_bridge_should_flatten_contiguous_tensor(void) {
  const size_t shape[] = {2, 2};
  FloatTensor *tensor = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(tensor);

  size_t idx00[] = {0, 0};
  size_t idx01[] = {0, 1};
  size_t idx10[] = {1, 0};
  size_t idx11[] = {1, 1};
  TEST_ASSERT_TRUE(st_set(tensor, idx00, 1.0f));
  TEST_ASSERT_TRUE(st_set(tensor, idx01, 2.0f));
  TEST_ASSERT_TRUE(st_set(tensor, idx10, 3.0f));
  TEST_ASSERT_TRUE(st_set(tensor, idx11, 4.0f));

  FloatVectorView view = st_as_vv_view(tensor);
  TEST_ASSERT_TRUE(vv_is_valid(&view));
  TEST_ASSERT_EQUAL_UINT32(4, (uint32_t)view.len);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 3.0f, vv_get(&view, 2));

  FloatVector *copy = st_to_sv(tensor);
  TEST_ASSERT_NOT_NULL(copy);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 4.0f, copy->values[3]);

  sv_destroy(copy);
  st_destroy(tensor);
}
