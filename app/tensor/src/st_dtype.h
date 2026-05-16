/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_dtype.h — Internal dtype conversion helpers.
 */

#ifndef ST_DTYPE_H
#define ST_DTYPE_H

#include "st.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
