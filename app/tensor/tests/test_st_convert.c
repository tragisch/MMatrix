/*
 * SPDX-License-Identifier: MIT
 */

#include "st_convert.h"

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_st_convert_st_from_sm_should_copy_shape_and_values(void) {
  float values[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  FloatMatrix *sm = sm_from_array_static(2, 3, values);
  TEST_ASSERT_NOT_NULL(sm);

  FloatTensor *st = st_from_sm(sm);
  TEST_ASSERT_NOT_NULL(st);
  TEST_ASSERT_EQUAL(2, st_tensor_ndim(st));
  TEST_ASSERT_EQUAL(2, st_tensor_shape(st)[0]);
  TEST_ASSERT_EQUAL(3, st_tensor_shape(st)[1]);
  TEST_ASSERT_EQUAL(6, st_tensor_numel(st));

  float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  size_t idx[2] = {0, 0};
  for (size_t i = 0; i < 6; ++i) {
    idx[0] = i / 3;
    idx[1] = i % 3;
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, expected[i], st_get(st, idx));
  }

  sm->values[0] = 99.0f;
  idx[0] = 0;
  idx[1] = 0;
  TEST_ASSERT_DOUBLE_WITHIN(1e-6, 1.0f, st_get(st, idx));

  sm_destroy(sm);
  st_destroy(st);
}

void test_st_convert_sm_from_st_should_copy_shape_and_values(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *st = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(st);

  float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  size_t idx[2] = {0, 0};
  for (size_t i = 0; i < 6; ++i) {
    idx[0] = i / 3;
    idx[1] = i % 3;
    TEST_ASSERT_TRUE(st_set(st, idx, values[i]));
  }

  FloatMatrix *sm = sm_from_st(st);
  TEST_ASSERT_NOT_NULL(sm);
  TEST_ASSERT_EQUAL(2, sm->rows);
  TEST_ASSERT_EQUAL(3, sm->cols);

  for (size_t i = 0; i < 6; ++i) {
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, values[i], sm->values[i]);
  }

  idx[0] = 0;
  idx[1] = 0;
  TEST_ASSERT_TRUE(st_set(st, idx, 99.0f));
  TEST_ASSERT_DOUBLE_WITHIN(1e-6, 1.0f, sm->values[0]);

  sm_destroy(sm);
  st_destroy(st);
}

void test_st_convert_sm_from_st_should_materialize_non_contiguous_2d_view(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *base = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(base);

  float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  size_t idx[2] = {0, 0};
  for (size_t i = 0; i < 6; ++i) {
    idx[0] = i / 3;
    idx[1] = i % 3;
    TEST_ASSERT_TRUE(st_set(base, idx, values[i]));
  }

  size_t perm[2] = {1, 0};
  FloatTensor *view = st_permute_view(base, perm);
  TEST_ASSERT_NOT_NULL(view);

  FloatMatrix *sm = sm_from_st(view);
  TEST_ASSERT_NOT_NULL(sm);
  TEST_ASSERT_EQUAL(3, sm->rows);
  TEST_ASSERT_EQUAL(2, sm->cols);

  float expected[3][2] = {
      {1.0f, 4.0f},
      {2.0f, 5.0f},
      {3.0f, 6.0f},
  };

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      TEST_ASSERT_DOUBLE_WITHIN(1e-6, expected[i][j], sm_get(sm, i, j));
    }
  }

  sm_destroy(sm);
  st_destroy(view);
  st_destroy(base);
}
