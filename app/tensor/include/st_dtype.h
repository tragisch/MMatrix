/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_dtype.h — Element data type enumeration and bf16↔f32 conversion.
 *
 * bfloat16 uses the same exponent range as float32 (8 bits) but only
 * 7 mantissa bits, giving ~3 decimal digits of precision.  Conversion
 * to/from float32 is a pure bit-shift operation — no lookup tables.
 *
 * On Apple Silicon the bulk converters use NEON SIMD for throughput.
 */

#ifndef ST_DTYPE_H
#define ST_DTYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Data-type tag                                                      */
/* ------------------------------------------------------------------ */

typedef enum StDtype {
  ST_DTYPE_F32 = 0,   /* IEEE 754 binary32  (4 bytes, default) */
  ST_DTYPE_BF16 = 1,  /* bfloat16           (2 bytes)          */
} StDtype;

/* Bytes per element for a given dtype. */
static inline size_t st_dtype_size(StDtype dtype) {
  switch (dtype) {
    case ST_DTYPE_BF16:
      return 2;
    case ST_DTYPE_F32:
    default:
      return 4;
  }
}

/* Human-readable name string. */
const char *st_dtype_name(StDtype dtype);

/* ------------------------------------------------------------------ */
/*  Scalar conversion                                                  */
/* ------------------------------------------------------------------ */

/* Convert one float32 → bfloat16 (round-to-nearest-even). */
uint16_t st_f32_to_bf16(float value);

/* Convert one bfloat16 → float32 (exact, zero-extends mantissa). */
float st_bf16_to_f32(uint16_t value);

/* ------------------------------------------------------------------ */
/*  Bulk conversion  (NEON-accelerated on aarch64)                     */
/* ------------------------------------------------------------------ */

/* Convert `n` float32 values to bfloat16 (round-to-nearest-even).
 * `src` and `dst` may NOT alias.                                     */
void st_f32_to_bf16_bulk(const float *src, uint16_t *dst, size_t n);

/* Convert `n` bfloat16 values to float32 (exact).
 * `src` and `dst` may NOT alias.                                     */
void st_bf16_to_f32_bulk(const uint16_t *src, float *dst, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* ST_DTYPE_H */
