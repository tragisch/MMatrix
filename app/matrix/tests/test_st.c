/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st.h"
#include "sm.h"

#include <stdint.h>

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
#ifndef TEST_ASSERT_EQUAL_INT
#define TEST_ASSERT_EQUAL_INT(expected, actual) \
  do {                                          \
    (void)(expected);                           \
    (void)(actual);                             \
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

#define EPSILON 1e-6f

void setUp(void) {}
void tearDown(void) {}

void test_st_numel_from_shape_should_compute_numel(void) {
  size_t shape[3] = {2, 3, 4};
  size_t numel = 0;

  bool ok = st_numel_from_shape(3, shape, &numel);

  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL_UINT32(24u, (uint32_t)numel);
}

void test_st_numel_from_shape_should_fail_on_invalid_shape(void) {
  size_t numel = 0;
  bool ok = st_numel_from_shape(0, NULL, &numel);
  TEST_ASSERT_FALSE(ok);
}

void test_st_compute_default_strides_should_be_row_major(void) {
  size_t shape[3] = {2, 3, 4};
  ptrdiff_t strides[3] = {0};

  bool ok = st_compute_default_strides(3, shape, strides);

  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL_INT(12, (int)strides[0]);
  TEST_ASSERT_EQUAL_INT(4, (int)strides[1]);
  TEST_ASSERT_EQUAL_INT(1, (int)strides[2]);
}

void test_st_create_should_initialize_tensor_metadata(void) {
  size_t shape[4] = {2, 3, 4, 5};
  FloatTensor *t = st_create(4, shape);

  TEST_ASSERT_NOT_NULL(t);
  TEST_ASSERT_EQUAL_UINT32(4u, (uint32_t)t->ndim);
  TEST_ASSERT_EQUAL_UINT32(120u, (uint32_t)t->numel);
  TEST_ASSERT_TRUE(t->owns_data);
  TEST_ASSERT_TRUE(st_is_contiguous(t));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->values[0]);

  st_destroy(t);
}

void test_st_create_should_fail_on_zero_dimension(void) {
  size_t shape[2] = {2, 0};
  FloatTensor *t = st_create(2, shape);
  TEST_ASSERT_NULL(t);
}

void test_st_set_get_should_access_by_multi_index(void) {
  size_t shape[3] = {2, 3, 4};
  FloatTensor *t = st_create(3, shape);
  TEST_ASSERT_NOT_NULL(t);

  size_t idx[3] = {1, 2, 3};
  bool ok = st_set(t, idx, 7.25f);

  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 7.25f, st_get(t, idx));

  size_t bad_idx[3] = {2, 0, 0};
  TEST_ASSERT_FALSE(st_set(t, bad_idx, 1.0f));

  st_destroy(t);
}

void test_st_reshape_should_keep_data_when_numel_matches(void) {
  size_t shape[2] = {2, 6};
  FloatTensor *t = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(t);

  for (size_t i = 0; i < t->numel; ++i) {
    t->values[i] = (float)(i + 1);
  }

  size_t new_shape[3] = {3, 2, 2};
  bool ok = st_reshape(t, 3, new_shape);

  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL_UINT32(3u, (uint32_t)t->ndim);
  TEST_ASSERT_TRUE(st_is_contiguous(t));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, t->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 12.0f, t->values[11]);

  size_t invalid_shape[2] = {5, 5};
  TEST_ASSERT_FALSE(st_reshape(t, 2, invalid_shape));

  st_destroy(t);
}

void test_st_view_should_reference_base_data_with_offset(void) {
  size_t base_shape[2] = {3, 4};
  FloatTensor *base = st_create(2, base_shape);
  TEST_ASSERT_NOT_NULL(base);

  for (size_t i = 0; i < base->numel; ++i) {
    base->values[i] = (float)i;
  }

  size_t view_shape[2] = {2, 4};
  ptrdiff_t view_strides[2] = {4, 1};
  FloatTensor *view = st_view(base, 2, view_shape, view_strides, 4);

  TEST_ASSERT_NOT_NULL(view);
  TEST_ASSERT_FALSE(view->owns_data);

  size_t idx0[2] = {0, 0};
  size_t idx1[2] = {1, 3};
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, st_get(view, idx0));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 11.0f, st_get(view, idx1));

  st_destroy(view);
  st_destroy(base);
}

void test_st_permute_view_should_swap_dimensions_without_copy(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *t = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(t);

  float data[] = {1, 2, 3, 4, 5, 6};
  for (size_t i = 0; i < 6; ++i) {
    t->values[i] = data[i];
  }

  size_t perm[2] = {1, 0};
  FloatTensor *p = st_permute_view(t, perm);
  TEST_ASSERT_NOT_NULL(p);

  TEST_ASSERT_EQUAL_UINT32(3u, (uint32_t)p->shape[0]);
  TEST_ASSERT_EQUAL_UINT32(2u, (uint32_t)p->shape[1]);

  size_t i01[2] = {0, 1};  // old(1,0)
  size_t i21[2] = {2, 1};  // old(1,2)
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, st_get(p, i01));
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, st_get(p, i21));

  st_destroy(p);
  st_destroy(t);
}

void test_st_clone_should_materialize_non_contiguous_view(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *t = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(t);

  float data[] = {1, 2, 3, 4, 5, 6};
  for (size_t i = 0; i < 6; ++i) {
    t->values[i] = data[i];
  }

  size_t perm[2] = {1, 0};
  FloatTensor *p = st_permute_view(t, perm);
  TEST_ASSERT_NOT_NULL(p);

  FloatTensor *c = st_clone(p);
  TEST_ASSERT_NOT_NULL(c);
  TEST_ASSERT_TRUE(st_is_contiguous(c));

  float expected[] = {1, 4, 2, 5, 3, 6};
  for (size_t i = 0; i < 6; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i], c->values[i]);
  }

  st_destroy(c);
  st_destroy(p);
  st_destroy(t);
}

void test_st_create_with_data_without_ownership_should_not_take_free_responsibility(
    void) {
  size_t shape[2] = {2, 2};
  float raw[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  FloatTensor *t = st_create_with_data(2, shape, raw, 4, false);
  TEST_ASSERT_NOT_NULL(t);
  TEST_ASSERT_FALSE(t->owns_data);

  size_t idx[2] = {1, 1};
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, st_get(t, idx));

  st_destroy(t);
}

void test_st_as_sm_view_should_expose_contiguous_2d_tensor_without_copy(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *t = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(t);

  for (size_t i = 0; i < t->numel; ++i) {
    t->values[i] = (float)(i + 1);
  }

  FloatMatrix view = {0};
  bool ok = st_as_sm_view(t, &view);

  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL(2, view.rows);
  TEST_ASSERT_EQUAL(3, view.cols);
  TEST_ASSERT_EQUAL(6, view.capacity);
  TEST_ASSERT_TRUE(view.values == t->values);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, view.values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, view.values[5]);

  st_destroy(t);
}

void test_st_as_sm_view_should_fail_for_non_2d_or_non_contiguous_tensor(void) {
  size_t shape3d[3] = {2, 2, 2};
  FloatTensor *t3 = st_create(3, shape3d);
  TEST_ASSERT_NOT_NULL(t3);

  FloatMatrix view = {0};
  TEST_ASSERT_FALSE(st_as_sm_view(t3, &view));
  st_destroy(t3);

  size_t shape2d[2] = {2, 3};
  FloatTensor *t2 = st_create(2, shape2d);
  TEST_ASSERT_NOT_NULL(t2);

  size_t perm[2] = {1, 0};
  FloatTensor *p = st_permute_view(t2, perm);
  TEST_ASSERT_NOT_NULL(p);
  TEST_ASSERT_FALSE(st_as_sm_view(p, &view));

  st_destroy(p);
  st_destroy(t2);
}
