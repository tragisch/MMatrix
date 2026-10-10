/* Bit-exactness tests for the f16 conversion API against a copy of nm_lib's
 * nm_halfprec.h reference. */

#include "st_f16.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define UNITY_INCLUDE_FLOAT
#define TEST_CASE(...)
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

/* ---- Reference (verbatim copy of nm_halfprec.h) ------------------------- */

static uint32_t nm_bits_of(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static float nm_float_of(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint16_t nm_f32_to_f16(float value) {
  uint32_t bits = nm_bits_of(value);
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000U);
  int32_t exponent = (int32_t)((bits >> 23) & 0xffU);
  uint32_t mantissa = bits & 0x007fffffU;
  if (exponent == 0xffU)
    return (uint16_t)(sign | 0x7c00U | (mantissa ? 0x0200U : 0U));
  int32_t half_exponent = exponent - 127 + 15;
  if (half_exponent >= 0x1f)
    return (uint16_t)(sign | 0x7c00U);
  if (half_exponent <= 0) {
    if (half_exponent < -10)
      return sign;
    uint32_t full = mantissa | 0x00800000U;
    uint32_t shift = (uint32_t)(14 - half_exponent);
    uint32_t half = full >> shift;
    uint32_t remainder = full & ((1U << shift) - 1U);
    uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (half & 1U)))
      half += 1U;
    return (uint16_t)(sign | half);
  }
  uint32_t half = ((uint32_t)half_exponent << 10) | (mantissa >> 13);
  uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half & 1U)))
    half += 1U;
  return (uint16_t)(sign | half);
}

static float nm_f16_to_f32(uint16_t bits) {
  uint32_t sign = (uint32_t)(bits & 0x8000U) << 16;
  uint32_t exponent = (bits >> 10) & 0x1fU;
  uint32_t mantissa = bits & 0x03ffU;
  uint32_t out;
  if (exponent == 0 && mantissa == 0) {
    out = sign;
  } else if (exponent == 0) {
    int32_t e = 127 - 15 + 1;
    while (!(mantissa & 0x0400U)) {
      mantissa <<= 1;
      --e;
    }
    out = sign | ((uint32_t)e << 23) | ((mantissa & 0x03ffU) << 13);
  } else if (exponent == 0x1f) {
    out = sign | 0x7f800000U | (mantissa ? 0x00400000U | (mantissa << 13) : 0U);
  } else {
    out = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  return nm_float_of(out);
}

static float float_of_bits(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

/* ---- Scalar vs reference ------------------------------------------------- */

void test_st_f16_scalar_matches_reference_on_edge_table(void) {
  /* +/-0, smallest subnormal, largest finite, overflow, +/-inf, NaN
   * payloads, and halfway rounding cases. */
  static const uint32_t f32_patterns[] = {
      0x00000000u, 0x80000000u,                  /* +/-0 */
      0x33ff0000u, 0xb3ff0000u,                  /* near smallest subnormal */
      0x387f0000u, 0x38800000u,                  /* 1.0/2 boundary */
      0x477fe000u, 0x477ff000u,                  /* near largest finite */
      0x477ff800u, 0x47800000u, 0x47800800u,     /* overflow boundary */
      0x7f800000u, 0xff800000u,                  /* +/-inf */
      0x7f800001u, 0xff800001u,                  /* signaling NaN */
      0x7fc00000u, 0xffc00000u,                  /* quiet NaN */
      0x7f801000u, 0x7f987a56u, 0x7fffffffu,     /* NaN payloads */
      0x3f000000u, 0x40490fdbu, 0xc0490fdbu,     /* 0.5, pi, -pi */
      0x3f800100u, 0x3f800200u,                  /* halfway mantissas */
      0x35000000u, 0x34000000u, 0x33000000u,     /* subnormal region */
      0x4b7fffffu, 0x4b000001u,                  /* rounding edges */
  };
  for (size_t i = 0; i < sizeof(f32_patterns) / sizeof(f32_patterns[0]); ++i) {
    const float value = float_of_bits(f32_patterns[i]);
    TEST_ASSERT_EQUAL_UINT16(nm_f32_to_f16(value), st_f32_to_f16(value));
  }
  /* Halfway rounding: remainder exactly 0x1000, odd vs even half. */
  TEST_ASSERT_EQUAL_UINT16(nm_f32_to_f16(float_of_bits(0x40000100u)),
                           st_f32_to_f16(float_of_bits(0x40000100u)));
  TEST_ASSERT_EQUAL_UINT16(nm_f32_to_f16(float_of_bits(0x40006100u)),
                           st_f32_to_f16(float_of_bits(0x40006100u)));
}

void test_st_f16_scalar_matches_reference_on_all_f16_roundtrips(void) {
  /* Every f16 pattern through f32 and back must reproduce the reference
   * exactly (non-NaN patterns also round-trip to themselves). */
  for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
    const uint16_t h = (uint16_t)bits;
    const float f32 = st_f16_to_f32(h);
    TEST_ASSERT_EQUAL_UINT32(nm_bits_of(nm_f16_to_f32(h)), nm_bits_of(f32));
    TEST_ASSERT_EQUAL_UINT16(nm_f32_to_f16(f32), st_f32_to_f16(f32));
    const bool is_nan = (h & 0x7c00u) == 0x7c00u && (h & 0x03ffu) != 0u;
    if (!is_nan) {
      TEST_ASSERT_EQUAL_UINT16(h, st_f32_to_f16(f32));
    }
  }
}

void test_st_f16_scalar_matches_reference_on_f32_sweep(void) {
  /* Strided sweep over all 2^32 f32 bit patterns plus a dense sweep of
   * the conversion-critical exponent range (subnormal and overflow
   * boundaries of binary16). */
  for (uint64_t bits = 0; bits <= 0xffffffffu; bits += 0x0009e37u) {
    const float value = float_of_bits((uint32_t)bits);
    TEST_ASSERT_EQUAL_UINT16(nm_f32_to_f16(value), st_f32_to_f16(value));
  }
  for (uint32_t exponent = 100; exponent <= 160; ++exponent) {
    for (uint32_t mantissa = 0; mantissa <= 0x007fffffu; mantissa += 0x00049249u) {
      const uint32_t bits = (exponent << 23) | mantissa;
      const float value = float_of_bits(bits);
      TEST_ASSERT_EQUAL_UINT16(nm_f32_to_f16(value), st_f32_to_f16(value));
    }
  }
}

/* ---- Bulk vs scalar ------------------------------------------------------- */

void test_st_f16_bulk_matches_scalar_for_short_lengths(void) {
  float src[40];
  uint16_t dst[40];
  for (size_t n = 0; n <= 33; ++n) {
    for (size_t i = 0; i < 40; ++i) {
      src[i] = (float)((int)i - 20) * 0.6180339887f + (float)(i % 7) * 1e-4f;
      if (i % 11 == 0) {
        src[i] = 0.0f;
      }
    }
    st_f32_to_f16_bulk(src, dst, n);
    for (size_t i = 0; i < n; ++i) {
      TEST_ASSERT_EQUAL_UINT16(st_f32_to_f16(src[i]), dst[i]);
    }
  }
}

void test_st_f16_bulk_f32_to_f16_matches_scalar_including_nan(void) {
  /* The NEON path must blend the reference NaN pattern; include NaN
   * payloads, inf, subnormals, and overflow values. */
  float src[1024];
  uint16_t dst[1024];
  uint16_t ref[1024];
  size_t n = 0;
  static const uint32_t patterns[] = {
      0x7f800001u, 0x7fc00001u, 0x7fffffffu, 0xff801234u, 0x7f987654u,
      0x00000001u, 0x80000001u, 0x33ff0000u, 0x477ff800u, 0x47800000u,
  };
  for (size_t p = 0; p < sizeof(patterns) / sizeof(patterns[0]); ++p) {
    src[n++] = float_of_bits(patterns[p]);
    src[n++] = -float_of_bits(patterns[p]);
  }
  for (uint32_t bits = 0x33800000u; bits <= 0x48000000u; bits += 0x001ab551u) {
    src[n++] = float_of_bits(bits);
    if (n >= 1000) {
      break;
    }
  }
  while (n < 1024) {
    src[n++] = (float)n * 0.0009765625f;
  }

  st_f32_to_f16_bulk(src, dst, n);
  for (size_t i = 0; i < n; ++i) {
    ref[i] = st_f32_to_f16(src[i]);
  }
  TEST_ASSERT_EQUAL_MEMORY(ref, dst, n * sizeof(uint16_t));
  /* Also against the copied reference. */
  for (size_t i = 0; i < n; ++i) {
    TEST_ASSERT_EQUAL_UINT16(nm_f32_to_f16(src[i]), dst[i]);
  }
}

void test_st_f16_bulk_f16_to_f32_matches_scalar_all_patterns(void) {
  /* All 65536 patterns, converted in one bulk call in chunks that cross
   * the 4-lane blocks and tails. */
  uint16_t src[65536];
  for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
    src[bits] = (uint16_t)bits;
  }
  float dst[65536];
  st_f16_to_f32_bulk(src, dst, 65536);
  for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
    TEST_ASSERT_EQUAL_UINT32(nm_bits_of(st_f16_to_f32((uint16_t)bits)),
                             nm_bits_of(dst[bits]));
  }

  /* Short lengths including tails. */
  for (size_t n = 0; n <= 33; ++n) {
    float short_dst[40];
    st_f16_to_f32_bulk(src, short_dst, n);
    for (size_t i = 0; i < n; ++i) {
      TEST_ASSERT_EQUAL_UINT32(nm_bits_of(st_f16_to_f32(src[i])),
                               nm_bits_of(short_dst[i]));
    }
  }
}

void test_st_f16_nan_policy_is_documented_behavior(void) {
  /* NaN stays NaN, quiet bit set, sign preserved, payload dropped. */
  TEST_ASSERT_EQUAL_UINT16(0x7e00u, st_f32_to_f16(float_of_bits(0x7f800001u)));
  TEST_ASSERT_EQUAL_UINT16(0xfe00u, st_f32_to_f16(float_of_bits(0xff801234u)));
  TEST_ASSERT_EQUAL_UINT16(0x7c00u, st_f32_to_f16(INFINITY));
  TEST_ASSERT_EQUAL_UINT16(0xfc00u, st_f32_to_f16(-INFINITY));
  /* Sign of +/-0 survives, including underflow to zero. */
  TEST_ASSERT_EQUAL_UINT16(0x0000u, st_f32_to_f16(0.0f));
  TEST_ASSERT_EQUAL_UINT16(0x8000u, st_f32_to_f16(-0.0f));
  TEST_ASSERT_EQUAL_UINT16(0x8000u, st_f32_to_f16(-1e-30f));
  /* Overflow goes to infinity. */
  TEST_ASSERT_EQUAL_UINT16(0x7c00u, st_f32_to_f16(1e30f));
  TEST_ASSERT_EQUAL_UINT16(0xfc00u, st_f32_to_f16(-1e30f));
  /* Smallest subnormal round-trips exactly. */
  const float smallest = st_f16_to_f32(0x0001u);
  TEST_ASSERT_EQUAL_UINT16(0x0001u, st_f32_to_f16(smallest));
  /* Largest finite stays finite. */
  const float largest = st_f16_to_f32(0x7bffu);
  TEST_ASSERT_EQUAL_UINT16(0x7bffu, st_f32_to_f16(largest));
}

void test_st_f16_bulk_subnormals_survive_fast_math(void) {
  /* Flush-to-zero would zero the subnormal outputs of the NEON path; this
   * builds f32 values around every subnormal half, at and between the
   * rounding boundaries, and requires bit-exact bulk == scalar. */
  float src[8192];
  uint16_t dst[8192];
  uint16_t ref[8192];
  size_t n = 0;
  for (uint32_t h = 0x0001u; h <= 0x03ffu && n + 5 <= 8192; h += 1u) {
    /* f32 values whose halves round around h: exact value and the
     * values just below/at/above the halfway points. */
    const uint32_t base = h << 13;
    static const uint32_t offsets[] = {0x0000u, 0x0fffu, 0x1000u,
                                       0x1001u, 0x1fffu};
    for (size_t k = 0; k < 5; ++k) {
      src[n++] = float_of_bits(base + offsets[k]);
    }
  }
  /* Denormal-region f32 inputs that round to zero / smallest subnormal. */
  static const uint32_t extra[] = {
      0x33000000u, 0x33fffffdu, 0x34000000u, 0x337fffffu, 0x33800000u,
      0x33ffffffu, 0x33fff800u, 0x33fff7ffu, 0x33fff000u, 0x33efffffu,
  };
  for (size_t k = 0; k < sizeof(extra) / sizeof(extra[0]); ++k) {
    src[n++] = float_of_bits(extra[k]);
  }
  TEST_ASSERT_TRUE(n > 0);

  st_f32_to_f16_bulk(src, dst, n);
  for (size_t i = 0; i < n; ++i) {
    ref[i] = st_f32_to_f16(src[i]);
  }
  TEST_ASSERT_EQUAL_MEMORY(ref, dst, n * sizeof(uint16_t));
  /* And against the copied nm_lib reference. */
  for (size_t i = 0; i < n; ++i) {
    TEST_ASSERT_EQUAL_UINT16(nm_f32_to_f16(src[i]), dst[i]);
  }

  /* Full roundtrip through both bulk kernels for all 65536 patterns:
   * h -> f32 -> h' must be the identity for non-NaN patterns (NaN
   * payloads canonicalize, matching the scalar functions). */
  static uint16_t patterns[65536];
  static float expanded[65536];
  static uint16_t back[65536];
  for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
    patterns[bits] = (uint16_t)bits;
  }
  st_f16_to_f32_bulk(patterns, expanded, 65536);
  st_f32_to_f16_bulk(expanded, back, 65536);
  for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
    TEST_ASSERT_EQUAL_UINT16(st_f32_to_f16(expanded[bits]), back[bits]);
    const bool is_nan = (bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0u;
    if (!is_nan) {
      TEST_ASSERT_EQUAL_UINT16((uint16_t)bits, back[bits]);
    }
  }
}
