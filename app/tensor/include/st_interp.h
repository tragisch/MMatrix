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

/**
 * @brief Gather slices along one axis, indexed by int64 values.
 * @param table Source tensor, contiguous, f32 or bf16.
 * @param indices Index array of length @p count; negative indices count
 *        from the end of @p axis.
 * @param count Number of indices; must be greater than zero.
 * @param axis Gather axis, below the tensor rank.
 * @param out Output tensor, contiguous, same dtype as @p table, shape equal
 *        to the table shape with @p axis replaced by @p count.
 * @retval true Success.
 * @retval false Invalid input or an index out of range; nothing is written.
 *
 * @details Indices are validated before any copy, so an out-of-range index
 *          fails without partial output (unlike the nm_lib reference, which
 *          copies until it hits the bad index). Slices are copied as
 *          contiguous blocks with memcpy. Index tensors of higher rank are
 *          covered by flattening them, matching the reference's
 *          [outer, count, inner] layout.
 */
bool st_gather(const FloatTensor *table, const int64_t *indices, size_t count,
               size_t axis, FloatTensor *out);

/**
 * @brief Gather for raw buffers of any element size.
 * @param table Source buffer, row-major, shape @p table_shape.
 * @param table_ndim Rank of @p table, `[1, 8]`.
 * @param table_shape Shape of @p table, all dimensions non-zero.
 * @param indices Index array of length @p count, negative counts from the
 *        end of @p axis (see @ref st_gather).
 * @param count Number of indices, greater than zero.
 * @param axis Gather axis, below @p table_ndim.
 * @param elem_bytes Element size in bytes (4 for f32/bf16-packed, 8 for
 *        int64, 1 for bool).
 * @param out Output buffer with `outer * count * inner` elements.
 * @retval true Success.
 * @retval false Invalid input or an index out of range; nothing is written.
 *
 * @details nm_lib gathers int64 and bool buffers too; the tensor API only
 *          has float dtypes, so this raw variant carries the element size.
 */
bool st_gather_raw(const void *table, size_t table_ndim,
                   const size_t *table_shape, const int64_t *indices,
                   size_t count, size_t axis, size_t elem_bytes, void *out);

/**
 * @brief Materialize a permuted copy into a prepared output tensor.
 * @param src Source tensor, contiguous, f32 or bf16.
 * @param perm Permutation where output axis `i` reads input axis
 *        `perm[i]` (same convention as @ref st_permute and nm_lib's
 *        run_transpose).
 * @param out Output tensor, contiguous, same dtype as @p src, shape equal
 *        to the permuted source shape.
 * @retval true Success.
 * @retval false Invalid input; nothing is written.
 *
 * @details @ref st_permute only creates views; the interpreter writes into
 *          pooled buffers, so it needs this materializing variant.
 */
bool st_permute_copy_into(const FloatTensor *src, const size_t *perm,
                          FloatTensor *out);

/**
 * @brief Materializing permute for raw buffers of any element size.
 * @param src Source buffer, row-major, shape @p shape.
 * @param ndim Rank, `[1, 8]`.
 * @param shape Source shape, all dimensions non-zero.
 * @param perm Permutation array where output axis `i` reads input axis
 *        `perm[i]`.
 * @param elem_bytes Element size in bytes.
 * @param out Output buffer with the permuted shape's element count.
 * @retval true Success.
 * @retval false Invalid input.
 *
 * @details Covers nm_lib's run_transpose for int64 and bool buffers, which
 *          the float-only tensor API cannot express.
 */
bool st_permute_copy_into_bytes(const void *src, size_t ndim,
                                const size_t *shape, const size_t *perm,
                                size_t elem_bytes, void *out);

#endif  // ST_INTERP_H
