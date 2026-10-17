/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_dtype.c — bf16↔f32 conversion implementation.
 *
 * Scalar functions use round-to-nearest-even (RNE) with correct NaN
 * handling.  Bulk functions use NEON SIMD on aarch64 for throughput;
 * NaN payloads in the bulk path may differ from the scalar path but
 * the result is always a valid NaN.
 */

#include "st_dtype.h"

#include <string.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define ST_HAS_NEON 1
#else
#define ST_HAS_NEON 0
#endif

/* ------------------------------------------------------------------ */
/*  Name helper                                                        */
/* ------------------------------------------------------------------ */

const char *st_dtype_name(StDtype dtype) {
  switch (dtype) {
    case ST_DTYPE_F32:
      return "float32";
    case ST_DTYPE_BF16:
      return "bfloat16";
    default:
      return "unknown";
  }
}

/* ------------------------------------------------------------------ */
/*  Scalar f32 → bf16  (round-to-nearest-even)                        */
/* ------------------------------------------------------------------ */

uint16_t st_f32_to_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));

  /* NaN → preserve as quiet NaN in bf16 */
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
    return (uint16_t)((bits >> 16) | 0x0040u);
  }

  /* Round-to-nearest-even (RNE):
   *   lsb        = bit 16 of the float  (will become LSB of bf16)
   *   round bit  = bit 15               (0.5 ULP of bf16)
   *   sticky     = bits 14..0           (< 0.5 ULP)
   *
   * Adding (0x7FFF + lsb) implements RNE in one step:
   *   - round bit 0          → no carry  (round down)
   *   - round bit 1, sticky≠0 → carry    (round up)
   *   - round bit 1, sticky=0 → carry iff lsb=1 (tie → even)
   */
  uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;

  return (uint16_t)(bits >> 16);
}

/* ------------------------------------------------------------------ */
/*  Scalar bf16 → f32  (exact)                                        */
/* ------------------------------------------------------------------ */

float st_bf16_to_f32(uint16_t value) {
  uint32_t bits = (uint32_t)value << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

/* ------------------------------------------------------------------ */
/*  Bulk f32 → bf16                                                    */
/* ------------------------------------------------------------------ */

void st_f32_to_bf16_bulk(const float *src, uint16_t *dst, size_t n) {
  if (!src || !dst || n == 0) return;

  size_t i = 0;

#if ST_HAS_NEON
  /* Process 8 floats per iteration (two 4-wide NEON lanes). */
  const uint32x4_t one = vdupq_n_u32(1u);
  const uint32x4_t bias = vdupq_n_u32(0x7FFFu);

  for (; i + 8 <= n; i += 8) {
    /* Lane 0: elements [i .. i+3] */
    uint32x4_t b0 = vreinterpretq_u32_f32(vld1q_f32(src + i));
    uint32x4_t lsb0 = vandq_u32(vshrq_n_u32(b0, 16), one);
    b0 = vaddq_u32(b0, vaddq_u32(bias, lsb0));
    uint16x4_t r0 = vshrn_n_u32(b0, 16);

    /* Lane 1: elements [i+4 .. i+7] */
    uint32x4_t b1 = vreinterpretq_u32_f32(vld1q_f32(src + i + 4));
    uint32x4_t lsb1 = vandq_u32(vshrq_n_u32(b1, 16), one);
    b1 = vaddq_u32(b1, vaddq_u32(bias, lsb1));
    uint16x4_t r1 = vshrn_n_u32(b1, 16);

    /* Combine into 8-wide uint16 and store. */
    vst1q_u16(dst + i, vcombine_u16(r0, r1));
  }

  /* Process remaining 4-wide chunk. */
  if (i + 4 <= n) {
    uint32x4_t b = vreinterpretq_u32_f32(vld1q_f32(src + i));
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(b, 16), one);
    b = vaddq_u32(b, vaddq_u32(bias, lsb));
    uint16x4_t r = vshrn_n_u32(b, 16);
    vst1_u16(dst + i, r);
    i += 4;
  }
#endif

  /* Scalar tail (or full fallback on non-NEON). */
  for (; i < n; ++i) {
    dst[i] = st_f32_to_bf16(src[i]);
  }
}

/* ------------------------------------------------------------------ */
/*  Bulk bf16 → f32                                                    */
/* ------------------------------------------------------------------ */

void st_bf16_to_f32_bulk(const uint16_t *src, float *dst, size_t n) {
  if (!src || !dst || n == 0) return;

  size_t i = 0;

#if ST_HAS_NEON
  /* Process 8 bf16 values per iteration. */
  for (; i + 8 <= n; i += 8) {
    /* Load 8 × uint16 → split into two 4-wide uint16 halves. */
    uint16x8_t h = vld1q_u16(src + i);
    uint16x4_t h0 = vget_low_u16(h);
    uint16x4_t h1 = vget_high_u16(h);

    /* Widen to uint32 and shift left 16 → float32 bit pattern. */
    uint32x4_t b0 = vshll_n_u16(h0, 16);
    uint32x4_t b1 = vshll_n_u16(h1, 16);

    vst1q_f32(dst + i, vreinterpretq_f32_u32(b0));
    vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(b1));
  }

  /* 4-wide tail. */
  if (i + 4 <= n) {
    uint16x4_t h = vld1_u16(src + i);
    uint32x4_t b = vshll_n_u16(h, 16);
    vst1q_f32(dst + i, vreinterpretq_f32_u32(b));
    i += 4;
  }
#endif

  /* Scalar tail. */
  for (; i < n; ++i) {
    dst[i] = st_bf16_to_f32(src[i]);
  }
}
