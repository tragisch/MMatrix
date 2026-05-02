/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "st_dtype.h"

#include "st.h"
#include "st_pool.h"
#include "st_batchnorm.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 6

/* Support for Meta Test Rig */
#define TEST_CASE(...)

#if __has_include("unity.h")
#include "unity.h"
#endif

#ifndef TEST_ASSERT_NOT_NULL
#define TEST_ASSERT_NOT_NULL(v) (void)(v)
#endif
#ifndef TEST_ASSERT_EQUAL
#define TEST_ASSERT_EQUAL(e, a) (void)(e); (void)(a)
#endif
#ifndef TEST_ASSERT_EQUAL_UINT16
#define TEST_ASSERT_EQUAL_UINT16(e, a) (void)(e); (void)(a)
#endif
#ifndef TEST_ASSERT_EQUAL_STRING
#define TEST_ASSERT_EQUAL_STRING(e, a) (void)(e); (void)(a)
#endif
#ifndef TEST_ASSERT_TRUE
#define TEST_ASSERT_TRUE(c) (void)(c)
#endif
#ifndef TEST_ASSERT_FLOAT_WITHIN
#define TEST_ASSERT_FLOAT_WITHIN(d, e, a) (void)(d); (void)(e); (void)(a)
#endif

void setUp(void) {}
void tearDown(void) {}

/* ================================================================== */
/*  st_dtype_size / st_dtype_name                                      */
/* ================================================================== */

void test_dtype_size_f32(void) {
  TEST_ASSERT_EQUAL(4u, st_dtype_size(ST_DTYPE_F32));
}

void test_dtype_size_bf16(void) {
  TEST_ASSERT_EQUAL(2u, st_dtype_size(ST_DTYPE_BF16));
}

void test_dtype_name_f32(void) {
  TEST_ASSERT_EQUAL_STRING("float32", st_dtype_name(ST_DTYPE_F32));
}

void test_dtype_name_bf16(void) {
  TEST_ASSERT_EQUAL_STRING("bfloat16", st_dtype_name(ST_DTYPE_BF16));
}

/* ================================================================== */
/*  Scalar f32 → bf16 → f32  round-trip                                */
/* ================================================================== */

void test_scalar_roundtrip_zero(void) {
  uint16_t h = st_f32_to_bf16(0.0f);
  TEST_ASSERT_EQUAL_UINT16(0x0000u, h);
  TEST_ASSERT_FLOAT_WITHIN(0.0f, 0.0f, st_bf16_to_f32(h));
}

void test_scalar_roundtrip_neg_zero(void) {
  uint16_t h = st_f32_to_bf16(-0.0f);
  TEST_ASSERT_EQUAL_UINT16(0x8000u, h);
  float back = st_bf16_to_f32(h);
  /* -0.0 == 0.0 by IEEE rule, but sign bit must be set. */
  uint32_t bits;
  memcpy(&bits, &back, sizeof(bits));
  TEST_ASSERT_TRUE((bits & 0x80000000u) != 0);
}

void test_scalar_roundtrip_one(void) {
  uint16_t h = st_f32_to_bf16(1.0f);
  TEST_ASSERT_EQUAL_UINT16(0x3F80u, h);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, st_bf16_to_f32(h));
}

void test_scalar_roundtrip_neg_one(void) {
  uint16_t h = st_f32_to_bf16(-1.0f);
  TEST_ASSERT_EQUAL_UINT16(0xBF80u, h);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, -1.0f, st_bf16_to_f32(h));
}

void test_scalar_roundtrip_pi_approx(void) {
  /* π ≈ 3.140625 in bf16 (about 3 decimal significant digits) */
  float pi = 3.14159265f;
  uint16_t h = st_f32_to_bf16(pi);
  float back = st_bf16_to_f32(h);
  TEST_ASSERT_FLOAT_WITHIN(0.02f, pi, back);
}

void test_scalar_roundtrip_large(void) {
  float v = 65504.0f;
  uint16_t h = st_f32_to_bf16(v);
  float back = st_bf16_to_f32(h);
  TEST_ASSERT_FLOAT_WITHIN(256.0f, v, back);  /* bf16 precision at this magnitude */
}

void test_scalar_roundtrip_small_positive(void) {
  float v = 1.0e-20f;
  uint16_t h = st_f32_to_bf16(v);
  float back = st_bf16_to_f32(h);
  /* bf16 has same exponent range as f32, so small values survive. */
  TEST_ASSERT_TRUE(back > 0.0f);
  TEST_ASSERT_FLOAT_WITHIN(v * 0.1f, v, back);
}

/* ================================================================== */
/*  Rounding behaviour                                                 */
/* ================================================================== */

void test_rne_tie_rounds_to_even(void) {
  /* Construct a float32 value exactly at the midpoint between two bf16
   * values.  The LSB of the bf16 result determines whether we round
   * up or down (round-to-nearest-even). */

  /* 1.0 in bf16 = 0x3F80.  Next bf16 value = 0x3F81.
   * Midpoint in f32 = 0x3F808000.                        */
  uint32_t mid_bits = 0x3F808000u;
  float mid;
  memcpy(&mid, &mid_bits, sizeof(mid));

  uint16_t h = st_f32_to_bf16(mid);
  /* LSB of 0x3F80 is 0 → tie rounds down (to even). */
  TEST_ASSERT_EQUAL_UINT16(0x3F80u, h);

  /* 1.0078125 (0x3F81) → midpoint = 0x3F818000 */
  uint32_t mid2_bits = 0x3F818000u;
  float mid2;
  memcpy(&mid2, &mid2_bits, sizeof(mid2));

  uint16_t h2 = st_f32_to_bf16(mid2);
  /* LSB of 0x3F81 is 1 → tie rounds up (to even = 0x3F82). */
  TEST_ASSERT_EQUAL_UINT16(0x3F82u, h2);
}

/* ================================================================== */
/*  Special values                                                     */
/* ================================================================== */

void test_scalar_inf(void) {
  uint16_t h_pos = st_f32_to_bf16(INFINITY);
  TEST_ASSERT_EQUAL_UINT16(0x7F80u, h_pos);
  TEST_ASSERT_TRUE(isinf(st_bf16_to_f32(h_pos)));

  uint16_t h_neg = st_f32_to_bf16(-INFINITY);
  TEST_ASSERT_EQUAL_UINT16(0xFF80u, h_neg);
  TEST_ASSERT_TRUE(isinf(st_bf16_to_f32(h_neg)));
}

void test_scalar_nan(void) {
  uint16_t h = st_f32_to_bf16(NAN);
  float back = st_bf16_to_f32(h);
  TEST_ASSERT_TRUE(isnan(back));

  /* Must be a quiet NaN (bit 6 of bf16 mantissa set = bit 22 of f32). */
  TEST_ASSERT_TRUE((h & 0x0040u) != 0);
}

/* ================================================================== */
/*  Bulk conversion                                                    */
/* ================================================================== */

void test_bulk_f32_to_bf16_basic(void) {
  float src[8] = {0.0f, 1.0f, -1.0f, 3.14f, 100.0f, -0.5f, 1e-10f, 42.0f};
  uint16_t dst[8] = {0};

  st_f32_to_bf16_bulk(src, dst, 8);

  for (size_t i = 0; i < 8; ++i) {
    uint16_t expected = st_f32_to_bf16(src[i]);
    TEST_ASSERT_EQUAL_UINT16(expected, dst[i]);
  }
}

void test_bulk_bf16_to_f32_basic(void) {
  float orig[8] = {0.0f, 1.0f, -1.0f, 3.14f, 100.0f, -0.5f, 1e-10f, 42.0f};
  uint16_t bf[8];
  for (size_t i = 0; i < 8; ++i) bf[i] = st_f32_to_bf16(orig[i]);

  float dst[8] = {0};
  st_bf16_to_f32_bulk(bf, dst, 8);

  for (size_t i = 0; i < 8; ++i) {
    float expected = st_bf16_to_f32(bf[i]);
    TEST_ASSERT_FLOAT_WITHIN(1e-30f, expected, dst[i]);
  }
}

void test_bulk_roundtrip_large(void) {
  /* Test with a size that exercises both NEON and scalar tail paths. */
  const size_t N = 1027;  /* 1024 (NEON) + 3 (scalar tail) */
  float *src = (float *)malloc(N * sizeof(float));
  uint16_t *bf = (uint16_t *)malloc(N * sizeof(uint16_t));
  float *dst = (float *)malloc(N * sizeof(float));

  for (size_t i = 0; i < N; ++i) {
    src[i] = (float)i * 0.1f - 50.0f;
  }

  st_f32_to_bf16_bulk(src, bf, N);
  st_bf16_to_f32_bulk(bf, dst, N);

  for (size_t i = 0; i < N; ++i) {
    /* bf16 has ~0.8% relative error for values > 1. */
    float expected = st_bf16_to_f32(st_f32_to_bf16(src[i]));
    TEST_ASSERT_FLOAT_WITHIN(1e-30f, expected, dst[i]);
  }

  free(src);
  free(bf);
  free(dst);
}

void test_bulk_null_args(void) {
  float src[4] = {1, 2, 3, 4};
  uint16_t dst[4] = {0};

  /* Should not crash. */
  st_f32_to_bf16_bulk(NULL, dst, 4);
  st_f32_to_bf16_bulk(src, NULL, 4);
  st_f32_to_bf16_bulk(src, dst, 0);
  st_bf16_to_f32_bulk(NULL, src, 4);
  st_bf16_to_f32_bulk(dst, NULL, 4);
  st_bf16_to_f32_bulk(dst, src, 0);

  TEST_ASSERT_TRUE(1);  /* reached without crash */
}

/* ================================================================== */
/*  bf16 tensor lifecycle                                              */
/* ================================================================== */

void test_create_bf16_tensor(void) {
  size_t shape[2] = {3, 4};
  FloatTensor *t = st_create_bf16(2, shape);

  TEST_ASSERT_NOT_NULL(t);
  TEST_ASSERT_EQUAL(2u, (unsigned)t->ndim);
  TEST_ASSERT_EQUAL(3u, (unsigned)t->shape[0]);
  TEST_ASSERT_EQUAL(4u, (unsigned)t->shape[1]);
  TEST_ASSERT_EQUAL(12u, (unsigned)t->numel);
  TEST_ASSERT_EQUAL(12u, (unsigned)t->capacity);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)t->dtype);
  TEST_ASSERT_TRUE(t->owns_data);

  st_destroy(t);
}

void test_bf16_tensor_half_memory(void) {
  size_t shape[2] = {64, 64};
  FloatTensor *f32 = st_create(2, shape);
  FloatTensor *bf16 = st_create_bf16(2, shape);

  TEST_ASSERT_NOT_NULL(f32);
  TEST_ASSERT_NOT_NULL(bf16);

  /* bf16 buffer should use roughly half the bytes. */
  TEST_ASSERT_TRUE(bf16->buf->size_bytes <= f32->buf->size_bytes / 2 + 64);

  st_destroy(f32);
  st_destroy(bf16);
}

void test_bf16_get_set_roundtrip(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *t = st_create_bf16(2, shape);
  TEST_ASSERT_NOT_NULL(t);

  size_t idx[2] = {1, 2};
  bool ok = st_set(t, idx, 3.14f);
  TEST_ASSERT_TRUE(ok);

  float val = st_get(t, idx);
  TEST_ASSERT_FLOAT_WITHIN(0.02f, 3.14f, val);

  /* Verify other elements are still zero. */
  size_t idx0[2] = {0, 0};
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, st_get(t, idx0));

  st_destroy(t);
}

void test_f32_to_bf16_tensor_conversion(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *f32 = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(f32);

  /* Fill with known values. */
  for (size_t i = 0; i < f32->numel; ++i) {
    f32->values[i] = (float)i * 1.5f;
  }

  FloatTensor *bf16 = st_to_bf16(f32);
  TEST_ASSERT_NOT_NULL(bf16);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)bf16->dtype);
  TEST_ASSERT_EQUAL(f32->numel, bf16->numel);

  /* Convert back to f32 and verify. */
  FloatTensor *back = st_to_f32(bf16);
  TEST_ASSERT_NOT_NULL(back);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_F32, (int)back->dtype);

  for (size_t i = 0; i < f32->numel; ++i) {
    float expected = st_bf16_to_f32(st_f32_to_bf16(f32->values[i]));
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected, back->values[i]);
  }

  st_destroy(f32);
  st_destroy(bf16);
  st_destroy(back);
}

void test_bf16_clone(void) {
  size_t shape[3] = {2, 3, 4};
  FloatTensor *orig = st_create_bf16(3, shape);
  TEST_ASSERT_NOT_NULL(orig);

  /* Write some values. */
  for (size_t i = 0; i < orig->numel; ++i) {
    size_t idx[3] = {i / 12, (i / 4) % 3, i % 4};
    st_set(orig, idx, (float)i * 0.5f);
  }

  FloatTensor *copy = st_clone(orig);
  TEST_ASSERT_NOT_NULL(copy);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)copy->dtype);
  TEST_ASSERT_EQUAL(orig->numel, copy->numel);

  /* Verify values match. */
  for (size_t i = 0; i < orig->numel; ++i) {
    size_t idx[3] = {i / 12, (i / 4) % 3, i % 4};
    TEST_ASSERT_FLOAT_WITHIN(0.5f, st_get(orig, idx), st_get(copy, idx));
  }

  /* Verify independence (modify original, copy unchanged). */
  size_t idx0[3] = {0, 0, 0};
  st_set(orig, idx0, 999.0f);
  TEST_ASSERT_FLOAT_WITHIN(0.5f, 0.0f, st_get(copy, idx0));

  st_destroy(orig);
  st_destroy(copy);
}

void test_to_f32_noop_for_f32(void) {
  size_t shape[1] = {4};
  FloatTensor *f32 = st_create(1, shape);
  TEST_ASSERT_NOT_NULL(f32);
  f32->values[0] = 42.0f;

  FloatTensor *result = st_to_f32(f32);
  TEST_ASSERT_NOT_NULL(result);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_F32, (int)result->dtype);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 42.0f, result->values[0]);

  /* Must be a copy, not the same pointer. */
  TEST_ASSERT_TRUE(result != f32);
  TEST_ASSERT_TRUE(result->values != f32->values);

  st_destroy(f32);
  st_destroy(result);
}

void test_to_bf16_noop_for_bf16(void) {
  size_t shape[1] = {4};
  FloatTensor *bf16 = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(bf16);
  size_t idx[1] = {0};
  st_set(bf16, idx, 42.0f);

  FloatTensor *result = st_to_bf16(bf16);
  TEST_ASSERT_NOT_NULL(result);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)result->dtype);

  TEST_ASSERT_TRUE(result != bf16);

  st_destroy(bf16);
  st_destroy(result);
}

/* ================================================================== */
/*  bf16 element-wise in-place ops                                     */
/* ================================================================== */

void test_bf16_inplace_add(void) {
  size_t shape[1] = {4};
  FloatTensor *a = st_create_bf16(1, shape);
  FloatTensor *b = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);

  size_t idx[1];
  for (size_t i = 0; i < 4; ++i) {
    idx[0] = i;
    st_set(a, idx, (float)(i + 1));  /* 1, 2, 3, 4 */
    st_set(b, idx, 10.0f);
  }

  bool ok = st_inplace_add(a, b);
  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)a->dtype);

  idx[0] = 0; TEST_ASSERT_FLOAT_WITHIN(0.1f, 11.0f, st_get(a, idx));
  idx[0] = 3; TEST_ASSERT_FLOAT_WITHIN(0.1f, 14.0f, st_get(a, idx));

  st_destroy(a);
  st_destroy(b);
}

void test_bf16_inplace_sub(void) {
  size_t shape[1] = {4};
  FloatTensor *a = st_create_bf16(1, shape);
  FloatTensor *b = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);

  size_t idx[1];
  for (size_t i = 0; i < 4; ++i) {
    idx[0] = i;
    st_set(a, idx, 10.0f);
    st_set(b, idx, (float)(i + 1));
  }

  bool ok = st_inplace_sub(a, b);
  TEST_ASSERT_TRUE(ok);

  idx[0] = 0; TEST_ASSERT_FLOAT_WITHIN(0.1f, 9.0f, st_get(a, idx));
  idx[0] = 3; TEST_ASSERT_FLOAT_WITHIN(0.1f, 6.0f, st_get(a, idx));

  st_destroy(a);
  st_destroy(b);
}

void test_bf16_inplace_scale(void) {
  size_t shape[1] = {4};
  FloatTensor *t = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(t);

  size_t idx[1];
  for (size_t i = 0; i < 4; ++i) {
    idx[0] = i;
    st_set(t, idx, (float)(i + 1));
  }

  bool ok = st_inplace_scale(t, 3.0f);
  TEST_ASSERT_TRUE(ok);

  idx[0] = 0; TEST_ASSERT_FLOAT_WITHIN(0.1f, 3.0f, st_get(t, idx));
  idx[0] = 3; TEST_ASSERT_FLOAT_WITHIN(0.1f, 12.0f, st_get(t, idx));

  st_destroy(t);
}

void test_bf16_inplace_elementwise_multiply(void) {
  size_t shape[1] = {4};
  FloatTensor *a = st_create_bf16(1, shape);
  FloatTensor *b = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(a);
  TEST_ASSERT_NOT_NULL(b);

  size_t idx[1];
  for (size_t i = 0; i < 4; ++i) {
    idx[0] = i;
    st_set(a, idx, (float)(i + 1));
    st_set(b, idx, 2.0f);
  }

  bool ok = st_inplace_elementwise_multiply(a, b);
  TEST_ASSERT_TRUE(ok);

  idx[0] = 0; TEST_ASSERT_FLOAT_WITHIN(0.1f, 2.0f, st_get(a, idx));
  idx[0] = 3; TEST_ASSERT_FLOAT_WITHIN(0.1f, 8.0f, st_get(a, idx));

  st_destroy(a);
  st_destroy(b);
}

void test_bf16_fill(void) {
  size_t shape[1] = {8};
  FloatTensor *t = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(t);

  bool ok = st_fill(t, 7.0f);
  TEST_ASSERT_TRUE(ok);

  size_t idx[1];
  for (size_t i = 0; i < 8; ++i) {
    idx[0] = i;
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 7.0f, st_get(t, idx));
  }

  st_destroy(t);
}

void test_bf16_fill_zero(void) {
  size_t shape[1] = {8};
  FloatTensor *t = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(t);

  /* Set non-zero first */
  st_fill(t, 5.0f);
  bool ok = st_fill(t, 0.0f);
  TEST_ASSERT_TRUE(ok);

  size_t idx[1];
  for (size_t i = 0; i < 8; ++i) {
    idx[0] = i;
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, st_get(t, idx));
  }

  st_destroy(t);
}

void test_bf16_apply_relu(void) {
  size_t shape[1] = {4};
  FloatTensor *t = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(t);

  size_t idx[1];
  float vals[] = {-2.0f, -0.5f, 0.0f, 3.0f};
  for (size_t i = 0; i < 4; ++i) {
    idx[0] = i;
    st_set(t, idx, vals[i]);
  }

  bool ok = st_apply_relu(t);
  TEST_ASSERT_TRUE(ok);

  idx[0] = 0; TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, st_get(t, idx));
  idx[0] = 1; TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, st_get(t, idx));
  idx[0] = 2; TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, st_get(t, idx));
  idx[0] = 3; TEST_ASSERT_FLOAT_WITHIN(0.1f, 3.0f, st_get(t, idx));

  st_destroy(t);
}

void test_bf16_sum_axes(void) {
  /* 2D [2, 3] bf16 tensor, reduce axis 0 → [3] */
  size_t shape[2] = {2, 3};
  FloatTensor *t = st_create_bf16(2, shape);
  TEST_ASSERT_NOT_NULL(t);

  /* Row 0: 1, 2, 3; Row 1: 4, 5, 6 */
  size_t idx[2];
  float count = 1.0f;
  for (size_t r = 0; r < 2; ++r) {
    for (size_t c = 0; c < 3; ++c) {
      idx[0] = r; idx[1] = c;
      st_set(t, idx, count);
      count += 1.0f;
    }
  }

  size_t axes[] = {0};
  FloatTensor *result = st_sum_axes(t, axes, 1);
  TEST_ASSERT_NOT_NULL(result);
  /* Result is f32 (promotion always produces f32). */
  TEST_ASSERT_EQUAL((int)ST_DTYPE_F32, (int)result->dtype);
  TEST_ASSERT_FLOAT_WITHIN(0.1f, 5.0f, result->values[0]);  /* 1+4 */
  TEST_ASSERT_FLOAT_WITHIN(0.1f, 7.0f, result->values[1]);  /* 2+5 */
  TEST_ASSERT_FLOAT_WITHIN(0.1f, 9.0f, result->values[2]);  /* 3+6 */

  st_destroy(t);
  st_destroy(result);
}

/* ================================================================== */
/*  bf16 view with offset                                              */
/* ================================================================== */

void test_bf16_view_with_offset(void) {
  size_t shape[1] = {8};
  FloatTensor *base = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(base);

  size_t idx[1];
  for (size_t i = 0; i < 8; ++i) {
    idx[0] = i;
    st_set(base, idx, (float)(i + 1));
  }

  /* View of elements [4..7] */
  size_t v_shape[1] = {4};
  ptrdiff_t v_strides[1] = {1};
  FloatTensor *view = st_view(base, 1, v_shape, v_strides, 4);
  TEST_ASSERT_NOT_NULL(view);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)view->dtype);

  idx[0] = 0; TEST_ASSERT_FLOAT_WITHIN(0.1f, 5.0f, st_get(view, idx));
  idx[0] = 1; TEST_ASSERT_FLOAT_WITHIN(0.1f, 6.0f, st_get(view, idx));
  idx[0] = 2; TEST_ASSERT_FLOAT_WITHIN(0.1f, 7.0f, st_get(view, idx));
  idx[0] = 3; TEST_ASSERT_FLOAT_WITHIN(0.1f, 8.0f, st_get(view, idx));

  st_destroy(view);
  st_destroy(base);
}

/* ================================================================== */
/*  bf16 padding                                                       */
/* ================================================================== */

void test_bf16_pad_nchw(void) {
  /* 4D [1,1,2,2] bf16 tensor, pad by 1 */
  size_t shape[4] = {1, 1, 2, 2};
  FloatTensor *input = st_create_bf16(4, shape);
  TEST_ASSERT_NOT_NULL(input);

  size_t idx[4];
  float val = 1.0f;
  for (size_t h = 0; h < 2; ++h) {
    for (size_t w = 0; w < 2; ++w) {
      idx[0] = 0; idx[1] = 0; idx[2] = h; idx[3] = w;
      st_set(input, idx, val);
      val += 1.0f;
    }
  }

  FloatTensor *padded = st_pad_nchw(input, 1, 1, 0.0f);
  TEST_ASSERT_NOT_NULL(padded);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)padded->dtype);
  TEST_ASSERT_EQUAL(4, (int)padded->shape[2]);  /* 2 + 2*1 */
  TEST_ASSERT_EQUAL(4, (int)padded->shape[3]);

  /* Check corner (should be 0) */
  idx[0] = 0; idx[1] = 0; idx[2] = 0; idx[3] = 0;
  TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, st_get(padded, idx));

  /* Check original data at (1,1) */
  idx[2] = 1; idx[3] = 1;
  TEST_ASSERT_FLOAT_WITHIN(0.1f, 1.0f, st_get(padded, idx));

  st_destroy(input);
  st_destroy(padded);
}

/* ================================================================== */
/*  bf16 maxpool2d forward                                             */
/* ================================================================== */

void test_bf16_maxpool2d(void) {
  /* [1,1,4,4] bf16 input, kernel 2x2, stride 2 → [1,1,2,2] */
  size_t in_shape[4] = {1, 1, 4, 4};
  FloatTensor *input = st_create_bf16(4, in_shape);
  TEST_ASSERT_NOT_NULL(input);

  /* Fill with ascending values */
  size_t idx[4];
  float val = 1.0f;
  for (size_t h = 0; h < 4; ++h) {
    for (size_t w = 0; w < 4; ++w) {
      idx[0] = 0; idx[1] = 0; idx[2] = h; idx[3] = w;
      st_set(input, idx, val);
      val += 1.0f;
    }
  }

  size_t out_shape[4] = {1, 1, 2, 2};
  FloatTensor *output = st_create_bf16(4, out_shape);
  TEST_ASSERT_NOT_NULL(output);

  bool ok = st_maxpool2d_nchw(input, 2, 2, 2, 2, 0, 0, output, NULL);
  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)output->dtype);

  /* Max of [1,2,5,6]=6, [3,4,7,8]=8, [9,10,13,14]=14, [11,12,15,16]=16 */
  idx[0] = 0; idx[1] = 0;
  idx[2] = 0; idx[3] = 0; TEST_ASSERT_FLOAT_WITHIN(0.1f, 6.0f, st_get(output, idx));
  idx[2] = 0; idx[3] = 1; TEST_ASSERT_FLOAT_WITHIN(0.1f, 8.0f, st_get(output, idx));
  idx[2] = 1; idx[3] = 0; TEST_ASSERT_FLOAT_WITHIN(0.1f, 14.0f, st_get(output, idx));
  idx[2] = 1; idx[3] = 1; TEST_ASSERT_FLOAT_WITHIN(0.1f, 16.0f, st_get(output, idx));

  st_destroy(input);
  st_destroy(output);
}

void test_bf16_maxpool2d_should_reject_bf16_indices(void) {
  size_t in_shape[4] = {1, 1, 4, 4};
  size_t out_shape[4] = {1, 1, 2, 2};

  FloatTensor *input = st_create_bf16(4, in_shape);
  FloatTensor *output = st_create_bf16(4, out_shape);
  FloatTensor *indices = st_create_bf16(4, out_shape);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(indices);

  bool ok = st_maxpool2d_nchw(input, 2, 2, 2, 2, 0, 0, output, indices);
  TEST_ASSERT_TRUE(!ok);

  st_destroy(input);
  st_destroy(output);
  st_destroy(indices);
}

/* ================================================================== */
/*  bf16 avgpool2d forward                                             */
/* ================================================================== */

void test_bf16_avgpool2d(void) {
  /* [1,1,4,4] bf16 input, kernel 2x2, stride 2 → [1,1,2,2] */
  size_t in_shape[4] = {1, 1, 4, 4};
  FloatTensor *input = st_create_bf16(4, in_shape);
  TEST_ASSERT_NOT_NULL(input);

  size_t idx[4];
  float val = 1.0f;
  for (size_t h = 0; h < 4; ++h) {
    for (size_t w = 0; w < 4; ++w) {
      idx[0] = 0; idx[1] = 0; idx[2] = h; idx[3] = w;
      st_set(input, idx, val);
      val += 1.0f;
    }
  }

  size_t out_shape[4] = {1, 1, 2, 2};
  FloatTensor *output = st_create_bf16(4, out_shape);
  TEST_ASSERT_NOT_NULL(output);

  bool ok = st_avgpool2d_nchw(input, 2, 2, 2, 2, 0, 0, output);
  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)output->dtype);

  /* Avg of [1,2,5,6]=3.5, [3,4,7,8]=5.5, [9,10,13,14]=11.5, [11,12,15,16]=13.5 */
  idx[0] = 0; idx[1] = 0;
  idx[2] = 0; idx[3] = 0; TEST_ASSERT_FLOAT_WITHIN(0.2f, 3.5f, st_get(output, idx));
  idx[2] = 0; idx[3] = 1; TEST_ASSERT_FLOAT_WITHIN(0.2f, 5.5f, st_get(output, idx));
  idx[2] = 1; idx[3] = 0; TEST_ASSERT_FLOAT_WITHIN(0.2f, 11.5f, st_get(output, idx));
  idx[2] = 1; idx[3] = 1; TEST_ASSERT_FLOAT_WITHIN(0.2f, 13.5f, st_get(output, idx));

  st_destroy(input);
  st_destroy(output);
}

/* ================================================================== */
/*  bf16 batchnorm2d forward                                           */
/* ================================================================== */

void test_bf16_batchnorm2d_forward(void) {
  /* [1,2,2,2] bf16 input, 2 channels */
  size_t shape4[4] = {1, 2, 2, 2};
  size_t shape1[1] = {2};

  FloatTensor *input = st_create_bf16(4, shape4);
  FloatTensor *output = st_create_bf16(4, shape4);
  FloatTensor *mean = st_create(1, shape1);
  FloatTensor *var = st_create(1, shape1);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);

  /* Fill channel 0 with 1.0, channel 1 with 2.0 */
  size_t idx[4];
  for (size_t h = 0; h < 2; ++h) {
    for (size_t w = 0; w < 2; ++w) {
      idx[0] = 0; idx[1] = 0; idx[2] = h; idx[3] = w;
      st_set(input, idx, 1.0f);
      idx[1] = 1;
      st_set(input, idx, 2.0f);
    }
  }

  bool ok = st_batchnorm2d_forward(input, NULL, NULL, 1e-5f, output, mean,
                                   var);
  TEST_ASSERT_TRUE(ok);
  TEST_ASSERT_EQUAL((int)ST_DTYPE_BF16, (int)output->dtype);

  /* With constant input per channel, mean = input, var = 0,
   * output = (x - mean) / sqrt(var + eps) ≈ 0. */
  idx[0] = 0; idx[1] = 0; idx[2] = 0; idx[3] = 0;
  TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, st_get(output, idx));

  st_destroy(input);
  st_destroy(output);
  st_destroy(mean);
  st_destroy(var);
}

void test_bf16_batchnorm2d_forward_should_reject_bf16_mean_var(void) {
  size_t shape4[4] = {1, 2, 2, 2};
  size_t shape1[1] = {2};

  FloatTensor *input = st_create_bf16(4, shape4);
  FloatTensor *output = st_create_bf16(4, shape4);
  FloatTensor *mean = st_create_bf16(1, shape1);
  FloatTensor *var = st_create_bf16(1, shape1);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(output);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);

  bool ok = st_batchnorm2d_forward(input, NULL, NULL, 1e-5f, output, mean,
                                   var);
  TEST_ASSERT_TRUE(!ok);

  st_destroy(input);
  st_destroy(output);
  st_destroy(mean);
  st_destroy(var);
}

void test_bf16_batchnorm2d_backward_should_reject_bf16_mean_var(void) {
  size_t shape4[4] = {1, 2, 2, 2};
  size_t shape1[1] = {2};

  FloatTensor *input = st_create_bf16(4, shape4);
  FloatTensor *grad_output = st_create_bf16(4, shape4);
  FloatTensor *grad_input = st_create_bf16(4, shape4);
  FloatTensor *mean = st_create_bf16(1, shape1);
  FloatTensor *var = st_create_bf16(1, shape1);
  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(grad_input);
  TEST_ASSERT_NOT_NULL(mean);
  TEST_ASSERT_NOT_NULL(var);

  bool ok = st_batchnorm2d_backward(grad_output, input, mean, var, NULL,
                                    1e-5f, grad_input, NULL, NULL);
  TEST_ASSERT_TRUE(!ok);

  st_destroy(input);
  st_destroy(grad_output);
  st_destroy(grad_input);
  st_destroy(mean);
  st_destroy(var);
}

void test_bf16_maxpool2d_backward_should_reject_bf16_indices(void) {
  size_t shape4[4] = {1, 1, 2, 2};
  size_t grad_shape[4] = {1, 1, 4, 4};

  FloatTensor *grad_output = st_create_bf16(4, shape4);
  FloatTensor *indices = st_create_bf16(4, shape4);
  FloatTensor *grad_input = st_create_bf16(4, grad_shape);
  TEST_ASSERT_NOT_NULL(grad_output);
  TEST_ASSERT_NOT_NULL(indices);
  TEST_ASSERT_NOT_NULL(grad_input);

  bool ok = st_maxpool2d_backward_nchw(grad_output, indices, 4, 4, grad_input);
  TEST_ASSERT_TRUE(!ok);

  st_destroy(grad_output);
  st_destroy(indices);
  st_destroy(grad_input);
}

/* ================================================================== */
/*  bf16 global avgpool2d forward                                      */
/* ================================================================== */

void test_bf16_global_avgpool2d(void) {
  /* [1,1,2,2] → [1,1,1,1] */
  size_t in_shape[4] = {1, 1, 2, 2};
  FloatTensor *input = st_create_bf16(4, in_shape);
  TEST_ASSERT_NOT_NULL(input);

  size_t idx[4] = {0, 0, 0, 0};
  st_set(input, idx, 2.0f);
  idx[2] = 0; idx[3] = 1; st_set(input, idx, 4.0f);
  idx[2] = 1; idx[3] = 0; st_set(input, idx, 6.0f);
  idx[2] = 1; idx[3] = 1; st_set(input, idx, 8.0f);

  size_t out_shape[4] = {1, 1, 1, 1};
  FloatTensor *output = st_create_bf16(4, out_shape);
  TEST_ASSERT_NOT_NULL(output);

  bool ok = st_global_avgpool2d_nchw(input, output);
  TEST_ASSERT_TRUE(ok);

  idx[0] = 0; idx[1] = 0; idx[2] = 0; idx[3] = 0;
  TEST_ASSERT_FLOAT_WITHIN(0.2f, 5.0f, st_get(output, idx));  /* (2+4+6+8)/4 */

  st_destroy(input);
  st_destroy(output);
}
