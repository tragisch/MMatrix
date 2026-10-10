/**
 * @file st_interp.h
 * @brief Public API for graph-interpreter tensor kernels.
 *
 * These kernels mirror the scalar reference loops of the nm_lib graph
 * interpreter (broadcast binary ops, LayerNorm, ReduceMax, batched matmul,
 * gather, and materializing permute) so the interpreter can reuse them
 * instead of its own loops. All functions require contiguous f32 tensors;
 * bf16 tensors are promoted to f32 for the computation and rounded back
 * where the destination is bf16.
 */

/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_INTERP_H
#define ST_INTERP_H

#include <stddef.h>
#include <stdint.h>

#include "st.h"

/** @brief Element-wise binary operations for the broadcast kernels. */
typedef enum StBinOp {
  ST_BIN_ADD = 0,  /* out = a + b */
  ST_BIN_SUB = 1,  /* out = a - b */
  ST_BIN_MUL = 2,  /* out = a * b */
  ST_BIN_DIV = 3,  /* out = a / b (IEEE values for division by zero) */
} StBinOp;

/**
 * @brief Apply a binary op with NumPy-style broadcasting over any axes.
 * @param op Operation to apply.
 * @param a Left input tensor (contiguous, f32 or bf16).
 * @param b Right input tensor (contiguous, f32 or bf16).
 * @return New contiguous f32 tensor on success, or NULL on invalid input,
 *         non-broadcastable shapes, or allocation failure.
 *
 * @details Fast paths (matching the nm_lib reference) are used when both
 *          shapes are equal, when one side is a scalar, or when one operand
 *          is a shape suffix of the output (bias/scale vectors); everything
 *          else runs a strided walk with vectorized row kernels. Division
 *          by zero produces the usual IEEE values. The result is always
 *          f32; call @ref st_to_bf16 to round-trip bf16 inputs.
 */
FloatTensor *st_binary_broadcast(StBinOp op, const FloatTensor *a,
                                 const FloatTensor *b);

/**
 * @brief Apply a binary op with broadcasting into a prepared output tensor.
 * @param op Operation to apply.
 * @param a Left input tensor (contiguous, f32 or bf16).
 * @param b Right input tensor (contiguous, f32 or bf16).
 * @param out Output tensor; contiguous with the broadcast shape of @p a and
 *        @p b, f32 or bf16. Must not alias @p a or @p b.
 * @retval true Success.
 * @retval false Invalid input or shape mismatch; nothing is written.
 *
 * @details Companion of @ref st_binary_broadcast for callers that reuse a
 *          pooled output buffer. When @p out is bf16 the f32 result is
 *          rounded back to bf16 with the st conversion helpers.
 */
bool st_binary_broadcast_into(StBinOp op, const FloatTensor *a,
                               const FloatTensor *b, FloatTensor *out);

#endif  // ST_INTERP_H
