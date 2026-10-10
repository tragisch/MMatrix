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

/**
 * @brief Layer-normalize over the last @p norm_rank axes.
 * @param x Input tensor (contiguous, f32 or bf16).
 * @param gamma Optional per-element scale of the normalized group; NULL or a
 *        contiguous tensor with `group` elements (f32 or bf16).
 * @param beta Optional per-element shift; NULL or like @p gamma.
 * @param norm_rank Number of trailing axes to normalize over, `[1, ndim]`.
 * @param eps Variance epsilon added before the inverse square root.
 * @param out Output tensor, contiguous with the same shape as @p x; f32 or
 *        bf16. May alias @p x (in-place normalization per group).
 * @retval true Success.
 * @retval false Invalid input; nothing is written.
 *
 * @details Matches the nm_lib reference: mean via eight partial sums, then
 *          the centered values, the variance over the centered values with
 *          eight partial sums, and `1/sqrtf(var + eps)`, so results are
 *          bit-identical regardless of the OpenMP thread count. bf16 inputs
 *          are promoted to f32; a bf16 output is rounded back.
 */
bool st_layernorm(const FloatTensor *x, const FloatTensor *gamma,
                  const FloatTensor *beta, size_t norm_rank, float eps,
                  FloatTensor *out);

/**
 * @brief Maximum over the axes selected in @p axes_mask.
 * @param x Input tensor (contiguous, f32 or bf16).
 * @param axes_mask Bitmask; axis `i` is reduced when bit `i` is set. Bits
 *        at or above the tensor rank are rejected.
 * @param keepdims Keep reduced axes as size-1 axes in the output.
 * @return New contiguous f32 tensor on success, or NULL on error.
 *
 * @details NaN handling deviates deliberately from the nm_lib scalar loop
 *          (which silently drops NaNs under `x > max`): if any element of a
 *          reduction group is NaN, the output slot becomes the bit pattern
 *          of the first NaN in scan order, so errors are not swallowed.
 *          Finite results and -INFINITY empty regions follow the reference.
 */
FloatTensor *st_reduce_max(const FloatTensor *x, uint32_t axes_mask,
                           bool keepdims);

/**
 * @brief Maximum over selected axes into a prepared output tensor.
 * @param x Input tensor (contiguous, f32 or bf16).
 * @param axes_mask Bitmask of axes to reduce; see @ref st_reduce_max.
 * @param keepdims Keep reduced axes as size-1 axes in the output.
 * @param out Output tensor with the reduced shape, f32 or bf16.
 * @retval true Success.
 * @retval false Invalid input or shape mismatch; nothing is written.
 */
bool st_reduce_max_into(const FloatTensor *x, uint32_t axes_mask,
                        bool keepdims, FloatTensor *out);

/**
 * @brief Batched matrix product with broadcast over the leading axes.
 * @param a Left tensor, rank 2 to 4, contiguous, f32 or bf16; the last two
 *        axes are `[M, K]`.
 * @param b Right tensor, rank 2 to 4, contiguous, f32 or bf16; the last two
 *        axes are `[K, N]`.
 * @param c Output tensor with the broadcast batch shape plus `[M, N]`,
 *        contiguous, f32 or bf16. Must not alias @p a or @p b.
 * @retval true Success.
 * @retval false Invalid input or shape mismatch; nothing is written.
 *
 * @details Matches nm_lib's run_matmul: when @p b is a single `[K, N]`
 *          matrix, the batches of @p a stack into the rows of one GEMM;
 *          otherwise every batch index maps onto @p a and @p b via NumPy
 *          broadcasting of the leading axes and each batch runs through
 *          the same GEMM backend as @ref sm_gemm (Accelerate/OpenBLAS,
 *          MPS through the runtime backend). Small per-batch GEMMs are
 *          distributed over OpenMP only above a measured threshold so the
 *          BLAS threads are not oversubscribed. Accumulation order follows
 *          the backend, so results are compared with a relative tolerance,
 *          not bit-exactly.
 */
bool st_matmul_batched(const FloatTensor *a, const FloatTensor *b,
                       FloatTensor *c);

#endif  // ST_INTERP_H
