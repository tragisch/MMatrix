/*
 * SPDX-License-Identifier: MIT
 */

#include "sm.h"
#include "sv.h"
#include "vector_bridge.h"
#include "vv.h"

#include <stdint.h>

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 5
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_vv_should_expose_row_and_column_views(void) {
  FloatMatrix *mat = sm_create(2, 3);
  TEST_ASSERT_NOT_NULL(mat);
  mat->values[0] = 1.0f;
  mat->values[1] = 2.0f;
  mat->values[2] = 3.0f;
  mat->values[3] = 4.0f;
  mat->values[4] = 5.0f;
  mat->values[5] = 6.0f;

  FloatVectorView row = sm_row_view(mat, 1);
  FloatVectorView col = sm_col_view(mat, 1);

  TEST_ASSERT_TRUE(vv_is_valid(&row));
  TEST_ASSERT_TRUE(vv_is_valid(&col));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 4.0f, vv_get(&row, 0));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 5.0f, vv_get(&row, 1));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 2.0f, vv_get(&col, 0));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 5.0f, vv_get(&col, 1));

  FloatVector *materialized = vv_to_sv(&col);
  TEST_ASSERT_NOT_NULL(materialized);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 29.0f, vv_dot(&col, &col));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 2.0f, materialized->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 5.0f, materialized->values[1]);

  sv_destroy(materialized);
  sm_destroy(mat);
}

void test_vv_copy_from_sv_should_write_back(void) {
  float data[] = {1.0f, 2.0f, 3.0f};
  FloatVectorView view = vv_make(data, 3, 1);
  const float replacement[] = {7.0f, 8.0f, 9.0f};
  FloatVector *vec = sv_create_with_values(3, replacement);

  TEST_ASSERT_TRUE(vv_copy_from_sv(&view, vec));
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 7.0f, data[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 8.0f, data[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, 9.0f, data[2]);

  sv_destroy(vec);
}
