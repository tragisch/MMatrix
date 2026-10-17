/**
 * @file st_f16.h
 * @brief Public API for IEEE binary16 (half precision) conversion.
 *
 * The scalar functions reproduce nm_lib's nm_halfprec.h bit for bit:
 * round-to-nearest-even, exact subnormals, overflow to infinity, NaN kept
 * quiet (payload dropped, sign preserved), and the sign of ±0 preserved.
 * The bulk functions use NEON FCVT where available; NaN results are
 * corrected to the reference bit pattern because the hardware keeps the
 * payload.
 */

/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_F16_H
#define ST_F16_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert one f32 value to IEEE binary16 bits.
 * @param value Input value.
 * @return binary16 bit pattern (round to nearest even, overflow to
 *         infinity, subnormals exact, NaN quiet with sign preserved).
 */
uint16_t st_f32_to_f16(float value);

/**
 * @brief Convert IEEE binary16 bits to one f32 value.
 * @param bits binary16 bit pattern.
 * @return Exact f32 value (all binary16 values are exactly representable).
 */
float st_f16_to_f32(uint16_t bits);

/**
 * @brief Convert a contiguous f32 buffer to binary16 bits.
 * @param src Source buffer with @p n values.
 * @param dst Destination buffer with @p n uint16 slots.
 * @param n Number of elements.
 *
 * @details Bit-identical to @ref st_f32_to_f16 applied per element; uses
 *          NEON `vcvt_f16_f32` with a NaN correction when available.
 */
void st_f32_to_f16_bulk(const float *src, uint16_t *dst, size_t n);

/**
 * @brief Convert a contiguous binary16 buffer to f32.
 * @param src Source buffer with @p n uint16 slots.
 * @param dst Destination buffer with @p n values.
 * @param n Number of elements.
 *
 * @details Bit-identical to @ref st_f16_to_f32 applied per element; uses
 *          NEON `vcvt_f32_f16` when available.
 */
void st_f16_to_f32_bulk(const uint16_t *src, float *dst, size_t n);

#ifdef __cplusplus
}
#endif

#endif  // ST_F16_H
