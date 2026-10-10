/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* IEEE binary16 conversion, bit-identical to nm_lib's nm_halfprec.h
 * (round to nearest even, subnormals, overflow to inf, quiet NaN).
 *
 * The scalar paths are the reference algorithm; the bulk paths use NEON
 * FCVT where available. For f32 -> f16 the hardware NaN result keeps the
 * payload while the reference canonicalizes to sign | 0x7e00, so the bulk
 * path blends the reference bits into NaN lanes. */

#include "st_f16.h"

#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define ST_F16_HAVE_NEON 1
#endif

uint16_t st_f32_to_f16(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
  const int32_t exponent = (int32_t)((bits >> 23) & 0xffu);
  uint32_t mantissa = bits & 0x007fffffu;
  if (exponent == 0xff) {
    return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x0200u : 0u));
  }
  const int32_t half_exponent = exponent - 127 + 15;
  if (half_exponent >= 0x1f) {
    return (uint16_t)(sign | 0x7c00u);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }
    const uint32_t full = mantissa | 0x00800000u;
    const uint32_t shift = (uint32_t)(14 - half_exponent);
    uint32_t half = full >> shift;
    const uint32_t remainder = full & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (half & 1u))) {
      half += 1u;
    }
    return (uint16_t)(sign | half);
  }
  uint32_t half = ((uint32_t)half_exponent << 10) | (mantissa >> 13);
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
    half += 1u;
  }
  return (uint16_t)(sign | half);
}

float st_f16_to_f32(uint16_t bits) {
  const uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 0x1fu;
  uint32_t mantissa = bits & 0x03ffu;
  uint32_t out;
  if (exponent == 0 && mantissa == 0) {
    out = sign;
  } else if (exponent == 0) {
    int32_t e = 127 - 15 + 1;
    while ((mantissa & 0x0400u) == 0) {
      mantissa <<= 1;
      --e;
    }
    out = sign | ((uint32_t)e << 23) | ((mantissa & 0x03ffu) << 13);
  } else if (exponent == 0x1f) {
    out = sign | 0x7f800000u | (mantissa ? 0x00400000u | (mantissa << 13) : 0u);
  } else {
    out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float value;
  memcpy(&value, &out, sizeof(value));
  return value;
}

void st_f32_to_f16_bulk(const float *src, uint16_t *dst, size_t n) {
#if defined(ST_F16_HAVE_NEON)
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t value = vld1q_f32(src + i);
    const uint32x4_t bits = vreinterpretq_u32_f32(value);
    /* Hardware conversion; the reference NaN pattern differs (the
     * hardware keeps the payload, the reference drops it), so NaN lanes
     * are blended with sign | 0x7e00. */
    const float16x4_t converted = vcvt_f16_f32(value);
    const uint32x4_t abs_bits = vandq_u32(bits, vdupq_n_u32(0x7fffffffu));
    const uint32x4_t is_nan =
        vcgtq_u32(abs_bits, vdupq_n_u32(0x7f800000u));
    const uint32x4_t top = vshrq_n_u32(bits, 16);
    const uint16x4_t ref_nan = vorr_u16(
        vand_u16(vmovn_u32(top), vdup_n_u16(0x8000u)), vdup_n_u16(0x7e00u));
    const uint16x4_t mask = vreinterpret_u16_u32(vmovn_u32(is_nan));
    const uint16x4_t result = vbsl_u16(
        mask, ref_nan, vreinterpret_u16_f16(converted));
    vst1_u16(dst + i, result);
  }
  for (; i < n; ++i) {
    dst[i] = st_f32_to_f16(src[i]);
  }
#else
  for (size_t i = 0; i < n; ++i) {
    dst[i] = st_f32_to_f16(src[i]);
  }
#endif
}

void st_f16_to_f32_bulk(const uint16_t *src, float *dst, size_t n) {
#if defined(ST_F16_HAVE_NEON)
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float16x4_t value = vld1_u16(src + i);
    vst1q_f32(dst + i, vcvt_f32_f16(value));
  }
  for (; i < n; ++i) {
    dst[i] = st_f16_to_f32(src[i]);
  }
#else
  for (size_t i = 0; i < n; ++i) {
    dst[i] = st_f16_to_f32(src[i]);
  }
#endif
}
