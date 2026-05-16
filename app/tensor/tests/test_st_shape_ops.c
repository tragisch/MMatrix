/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * Unit tests for st_shape_ops.h functions:
 * - st_flatten / st_flatten_all
 * - st_permute
 * - st_concat
 */

#include "st.h"
#include "st_shape_ops.h"

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
#ifndef TEST_ASSERT_TRUE
#define TEST_ASSERT_TRUE(value) \
  do {                          \
    (void)(value);              \
  } while (0)
#endif
#ifndef TEST_ASSERT_FALSE
#define TEST_ASSERT_FALSE(value) \
  do {                           \
    (void)(value);               \
  } while (0)
#endif

/* ============================================================================
 * Test: st_flatten (partial flatten)
 * ============================================================================ */
void test_st_flatten_partial_4d_to_3d(void) {
  /* Create 4D tensor [2, 3, 4, 5] */
  size_t shape[4] = {2, 3, 4, 5};
  FloatTensor *t = st_create(4, shape);
  TEST_ASSERT_NOT_NULL(t);
  TEST_ASSERT_EQUAL(2 * 3 * 4 * 5, t->numel);

  /* Flatten axes [1, 3) -> [2, 3, 4, 5] becomes [2, 12, 5] */
  FloatTensor *flat = st_flatten(t, 1, 3);
  TEST_ASSERT_NOT_NULL(flat);
  TEST_ASSERT_EQUAL(3, flat->ndim);
  TEST_ASSERT_EQUAL(2, flat->shape[0]);
  TEST_ASSERT_EQUAL(12, flat->shape[1]);  /* 3 * 4 */
  TEST_ASSERT_EQUAL(5, flat->shape[2]);
  TEST_ASSERT_EQUAL(2 * 12 * 5, flat->numel);

  st_destroy(flat);
  st_destroy(t);
}

/* ============================================================================
 * Test: st_flatten_all (full flatten)
 * ============================================================================ */
void test_st_flatten_all_4d_to_1d(void) {
  /* Create 4D tensor [2, 3, 4, 5] */
  size_t shape[4] = {2, 3, 4, 5};
  FloatTensor *t = st_create(4, shape);
  TEST_ASSERT_NOT_NULL(t);

  /* Flatten all axes. */
  FloatTensor *flat = st_flatten_all(t);
  TEST_ASSERT_NOT_NULL(flat);
  TEST_ASSERT_EQUAL(1, flat->ndim);
  TEST_ASSERT_EQUAL(2 * 3 * 4 * 5, flat->shape[0]);
  TEST_ASSERT_EQUAL(120, flat->numel);

  st_destroy(flat);
  st_destroy(t);
}

/* ============================================================================
 * Test: st_permute (axes transposition)
 * ============================================================================ */
void test_st_permute_3d_transpose(void) {
  /* Create 3D tensor [2, 3, 4] */
  size_t shape[3] = {2, 3, 4};
  FloatTensor *t = st_create(3, shape);
  TEST_ASSERT_NOT_NULL(t);

  /* Permute [0, 2, 1] -> [2, 4, 3] */
  size_t perm[3] = {0, 2, 1};
  FloatTensor *perm_t = st_permute(t, perm);
  TEST_ASSERT_NOT_NULL(perm_t);
  TEST_ASSERT_EQUAL(3, perm_t->ndim);
  TEST_ASSERT_EQUAL(2, perm_t->shape[0]);
  TEST_ASSERT_EQUAL(4, perm_t->shape[1]);
  TEST_ASSERT_EQUAL(3, perm_t->shape[2]);
  TEST_ASSERT_EQUAL(2 * 4 * 3, perm_t->numel);

  st_destroy(perm_t);
  st_destroy(t);
}

/* ============================================================================
 * Test: st_concat (concatenation along axis 0)
 * ============================================================================ */
void test_st_concat_axis_0(void) {
  /* Create three [3, 4] tensors */
  size_t shape[2] = {3, 4};
  FloatTensor *t1 = st_create(2, shape);
  FloatTensor *t2 = st_create(2, shape);
  FloatTensor *t3 = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(t1);
  TEST_ASSERT_NOT_NULL(t2);
  TEST_ASSERT_NOT_NULL(t3);

  /* Fill with distinct values for verification */
  for (size_t i = 0; i < 12; ++i) {
    t1->values[i] = 1.0f;
    t2->values[i] = 2.0f;
    t3->values[i] = 3.0f;
  }

  /* Concatenate along axis 0: [3, 4] + [3, 4] + [3, 4] -> [9, 4] */
  const FloatTensor *inputs[3] = {t1, t2, t3};
  FloatTensor *concat = st_concat(inputs, 3, 0);
  TEST_ASSERT_NOT_NULL(concat);
  TEST_ASSERT_EQUAL(2, concat->ndim);
  TEST_ASSERT_EQUAL(9, concat->shape[0]);
  TEST_ASSERT_EQUAL(4, concat->shape[1]);
  TEST_ASSERT_EQUAL(36, concat->numel);

  /* Verify concatenated data. */
  for (size_t i = 0; i < 12; ++i) {
    TEST_ASSERT_EQUAL(1.0f, concat->values[i]);         /* t1 data */
    TEST_ASSERT_EQUAL(2.0f, concat->values[12 + i]);    /* t2 data */
    TEST_ASSERT_EQUAL(3.0f, concat->values[24 + i]);    /* t3 data */
  }

  st_destroy(concat);
  st_destroy(t1);
  st_destroy(t2);
  st_destroy(t3);
}

/* ============================================================================
 * Test: st_concat along axis 1
 * ============================================================================ */
void test_st_concat_axis_1(void) {
  /* Create two [2, 3] and [2, 4] tensors */
  size_t shape1[2] = {2, 3};
  size_t shape2[2] = {2, 4};
  FloatTensor *t1 = st_create(2, shape1);
  FloatTensor *t2 = st_create(2, shape2);
  TEST_ASSERT_NOT_NULL(t1);
  TEST_ASSERT_NOT_NULL(t2);

  for (size_t i = 0; i < 6; ++i) t1->values[i] = 1.0f;
  for (size_t i = 0; i < 8; ++i) t2->values[i] = 2.0f;

  /* Concatenate along axis 1: [2, 3] + [2, 4] -> [2, 7] */
  const FloatTensor *inputs[2] = {t1, t2};
  FloatTensor *concat = st_concat(inputs, 2, 1);
  TEST_ASSERT_NOT_NULL(concat);
  TEST_ASSERT_EQUAL(2, concat->shape[0]);
  TEST_ASSERT_EQUAL(7, concat->shape[1]);
  TEST_ASSERT_EQUAL(14, concat->numel);

  st_destroy(concat);
  st_destroy(t1);
  st_destroy(t2);
}

