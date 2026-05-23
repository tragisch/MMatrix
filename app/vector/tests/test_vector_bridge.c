/*
 * SPDX-License-Identifier: MIT
 */

#include "sm.h"
#include "st.h"
#include "sv.h"
#include "vector_bridge.h"
#include "vector_tensor_bridge.h"

#include <stdint.h>

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 5
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_vector_bridge_sm_matvec_and_outer_should_work(void) {
  FloatMatrix *mat = sm_create(2, 3);
  const float vec_values[] = {1.0f, 2.0f, 3.0f};
  const float rhs_values[] = {4.0f, 5.0f};
  FloatVector *vec = sv_create_with_values(3, vec_values);
  FloatVector *rhs = sv_create_with_values(2, rhs_values);
  TEST_ASSERT_NOT_NULL(mat);
  mat->values[0] = 1.0f;
  mat->values[1] = 0.0f;
  mat->values[2] = 2.0f;
  mat->values[3] = 0.0f;
  mat->values[4] = 1.0f;
  mat->values[5] = 3.0f;

  FloatVector *product = sm_matvec(mat, vec);
  FloatMatrix *outer = sv_outer_as_sm(rhs, vec);

  TEST_ASSERT_NOT_NULL(product);
  TEST_ASSERT_NOT_NULL(outer);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 7.0f, product->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 11.0f, product->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 4.0f, outer->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 10.0f, outer->values[4]);

  sm_destroy(outer);
  sv_destroy(product);
  sv_destroy(rhs);
  sv_destroy(vec);
  sm_destroy(mat);
}

void test_vector_bridge_tensor_view_should_flatten_contiguous_tensor(void) {
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
