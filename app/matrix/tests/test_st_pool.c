/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_pool.h"

#include <math.h>

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
#ifndef TEST_ASSERT_EQUAL
#define TEST_ASSERT_EQUAL(expected, actual) \
  do {                                      \
    (void)(expected);                       \
    (void)(actual);                         \
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

static FloatTensor *create_tensor_4d(size_t n, size_t c, size_t h, size_t w) {
  size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

/* ---- Output size ---- */

void test_st_pool2d_output_hw_should_compute_expected_shape(void) {
  size_t out_h = 0, out_w = 0;
  bool ok = st_pool2d_output_hw(4, 4, 2, 2, 2, 2, 0, 0, &out_h, &out_w);
  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL(2, out_h);
  TEST_ASSERT_EQUAL(2, out_w);
}

void test_st_pool2d_output_hw_with_padding(void) {
  size_t out_h = 0, out_w = 0;
  bool ok = st_pool2d_output_hw(4, 4, 3, 3, 1, 1, 1, 1, &out_h, &out_w);
  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL(4, out_h);
  TEST_ASSERT_EQUAL(4, out_w);
}

/* ---- Max Pool 2D Forward ---- */

void test_st_maxpool2d_nchw_basic(void) {
  /* input 1x1x4x4 */
  FloatTensor *input = create_tensor_4d(1, 1, 4, 4);
  TEST_ASSERT_NOT_NULL(input);

  float vals[] = {
      1,  2,  3,  4,   //
      5,  6,  7,  8,   //
      9,  10, 11, 12,  //
      13, 14, 15, 16   //
  };
  for (size_t i = 0; i < 16; ++i) {
    input->values[i] = vals[i];
  }

  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *indices = create_tensor_4d(1, 1, 2, 2);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(indices);

  bool ok = st_maxpool2d_nchw(input, 2, 2, 2, 2, 0, 0, output, indices);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, output->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 8.0f, output->values[1]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 14.0f, output->values[2]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 16.0f, output->values[3]);

  /* Check indices point to correct positions. */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, indices->values[0]);  /* row1,col1 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 7.0f, indices->values[1]);  /* row1,col3 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 13.0f, indices->values[2]); /* row3,col1 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 15.0f, indices->values[3]); /* row3,col3 */

  st_destroy(indices);
  st_destroy(output);
  st_destroy(input);
}

void test_st_maxpool2d_nchw_without_indices(void) {
  FloatTensor *input = create_tensor_4d(1, 1, 4, 4);
  TEST_ASSERT_NOT_NULL(input);
  for (size_t i = 0; i < 16; ++i) {
    input->values[i] = (float)(i + 1);
  }

  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);
  TEST_ASSERT_NOT_NULL(output);

  bool ok = st_maxpool2d_nchw(input, 2, 2, 2, 2, 0, 0, output, NULL);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 6.0f, output->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 16.0f, output->values[3]);

  st_destroy(output);
  st_destroy(input);
}

/* ---- Average Pool 2D Forward ---- */

void test_st_avgpool2d_nchw_basic(void) {
  FloatTensor *input = create_tensor_4d(1, 1, 4, 4);
  TEST_ASSERT_NOT_NULL(input);

  for (size_t i = 0; i < 16; ++i) {
    input->values[i] = (float)(i + 1);
  }

  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);
  TEST_ASSERT_NOT_NULL(output);

  bool ok = st_avgpool2d_nchw(input, 2, 2, 2, 2, 0, 0, output);
  TEST_ASSERT_TRUE(ok);

  /* average of {1,2,5,6} = 3.5 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.5f, output->values[0]);
  /* average of {3,4,7,8} = 5.5 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.5f, output->values[1]);
  /* average of {9,10,13,14} = 11.5 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 11.5f, output->values[2]);
  /* average of {11,12,15,16} = 13.5 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 13.5f, output->values[3]);

  st_destroy(output);
  st_destroy(input);
}

/* ---- Max Pool 2D Backward ---- */

void test_st_maxpool2d_backward_nchw_should_scatter_gradient(void) {
  /* Forward first to get indices. */
  FloatTensor *input = create_tensor_4d(1, 1, 4, 4);
  TEST_ASSERT_NOT_NULL(input);
  for (size_t i = 0; i < 16; ++i) {
    input->values[i] = (float)(i + 1);
  }

  FloatTensor *output = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *indices = create_tensor_4d(1, 1, 2, 2);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(indices);

  st_maxpool2d_nchw(input, 2, 2, 2, 2, 0, 0, output, indices);

  /* Backward: grad_output = all ones. */
  FloatTensor *grad_output = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *grad_input = create_tensor_4d(1, 1, 4, 4);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(grad_input);

  for (size_t i = 0; i < 4; ++i) {
    grad_output->values[i] = 1.0f;
  }

  bool ok =
      st_maxpool2d_backward_nchw(grad_output, indices, 4, 4, grad_input);
  TEST_ASSERT_TRUE(ok);

  /* Only max positions should have gradient 1, rest 0. */
  /* Max positions: (1,1)=idx5, (1,3)=idx7, (3,1)=idx13, (3,3)=idx15 */
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, grad_input->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, grad_input->values[5]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, grad_input->values[7]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, grad_input->values[8]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, grad_input->values[13]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, grad_input->values[15]);

  st_destroy(grad_input);
  st_destroy(grad_output);
  st_destroy(indices);
  st_destroy(output);
  st_destroy(input);
}

/* ---- Average Pool 2D Backward ---- */

void test_st_avgpool2d_backward_nchw_should_distribute_gradient(void) {
  FloatTensor *grad_output = create_tensor_4d(1, 1, 2, 2);
  FloatTensor *grad_input = create_tensor_4d(1, 1, 4, 4);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(grad_input);

  /* grad_output = all 1.0 */
  for (size_t i = 0; i < 4; ++i) {
    grad_output->values[i] = 1.0f;
  }

  bool ok = st_avgpool2d_backward_nchw(grad_output, 2, 2, 2, 2, 0, 0,
                                       grad_input);
  TEST_ASSERT_TRUE(ok);

  /* Each output gradient 1.0 distributed to 4 input positions → 0.25 each. */
  for (size_t i = 0; i < 16; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.25f, grad_input->values[i]);
  }

  st_destroy(grad_input);
  st_destroy(grad_output);
}

/* ---- Global Average Pool 2D ---- */

void test_st_global_avgpool2d_nchw_basic(void) {
  FloatTensor *input = create_tensor_4d(1, 2, 3, 3);
  TEST_ASSERT_NOT_NULL(input);

  /* Channel 0: all 1.0, channel 1: all 2.0. */
  for (size_t i = 0; i < 9; ++i) {
    input->values[i] = 1.0f;
  }
  for (size_t i = 9; i < 18; ++i) {
    input->values[i] = 2.0f;
  }

  FloatTensor *output = create_tensor_4d(1, 2, 1, 1);
  TEST_ASSERT_NOT_NULL(output);

  bool ok = st_global_avgpool2d_nchw(input, output);
  TEST_ASSERT_TRUE(ok);

  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, output->values[0]);
  TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, output->values[1]);

  st_destroy(output);
  st_destroy(input);
}

void test_st_global_avgpool2d_backward_nchw_basic(void) {
  FloatTensor *grad_output = create_tensor_4d(1, 1, 1, 1);
  FloatTensor *grad_input = create_tensor_4d(1, 1, 3, 3);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(grad_input);

  grad_output->values[0] = 9.0f;

  bool ok = st_global_avgpool2d_backward_nchw(grad_output, grad_input);
  TEST_ASSERT_TRUE(ok);

  /* 9.0 / 9 = 1.0 for each spatial position. */
  for (size_t i = 0; i < 9; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, grad_input->values[i]);
  }

  st_destroy(grad_input);
  st_destroy(grad_output);
}


