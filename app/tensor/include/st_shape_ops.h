/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_SHAPE_OPS_H
#define ST_SHAPE_OPS_H

#include <stddef.h>
#include "st.h"

/* ============================================================================
 * Shape Transformation Operations
 *
 * Convenience wrappers and new operations for tensor reshape, permute, 
 * flatten, and concatenation. These build on the core st_view() and other
 * primitives to provide a more user-friendly API for common shape manipulations.
 * ============================================================================ */

/**
 * Flatten a range of axes into a single axis.
 * 
 * Creates a view where axes [start_axis, end_axis) are merged into one.
 * For a 4D tensor [N, C, H, W]:
 *   - flatten(t, 1, 4) -> [N, C*H*W]  (flatten all but batch)
 *   - flatten(t, 0, 4) -> [N*C*H*W]   (flatten all)
 *   - flatten(t, 2, 4) -> [N, C, H*W] (flatten spatial)
 *
 * @param tensor Input tensor.
 * @param start_axis First axis to merge (0-indexed; inclusive).
 * @param end_axis Last axis to merge (exclusive; end_axis <= ndim).
 * @return New flattened view (no data copy), or NULL on error.
 *         Caller must st_destroy() the returned tensor.
 */
FloatTensor *st_flatten(const FloatTensor *tensor, size_t start_axis,
                        size_t end_axis);

/**
 * Flatten all axes into a 1D vector.
 *
 * Convenience for st_flatten(t, 0, t->ndim).
 *
 * @param tensor Input tensor.
 * @return New 1D view (no data copy), or NULL on error.
 *         Caller must st_destroy() the returned tensor.
 */
FloatTensor *st_flatten_all(const FloatTensor *tensor);

/**
 * Permute tensor axes.
 *
 * This is a convenience wrapper around st_permute_view() with additional
 * validation and optional inverse permutation support.
 *
 * @param tensor Input tensor.
 * @param axes Permutation array: axes[i] is the source axis for output axis i.
 *             Must be a valid permutation of [0, ndim).
 * @return New permuted view (no data copy), or NULL on error.
 *         Caller must st_destroy() the returned tensor.
 */
FloatTensor *st_permute(const FloatTensor *tensor, const size_t *axes);

/**
 * Reshape a contiguous tensor to a new shape.
 *
 * In-place reshape. Requires contiguous layout and matching element count.
 *
 * @param tensor Tensor to reshape (will be modified in-place).
 * @param new_ndim Number of dimensions in new shape.
 * @param new_shape Target shape array (length new_ndim).
 * @return true on success, false on error.
 *
 * NOTE: This is a convenience alias/wrapper for st_reshape() in st.h.
 *       See st_reshape() for detailed behavior and constraints.
 */
static inline bool st_reshape_to(FloatTensor *tensor, size_t new_ndim,
                                  const size_t *new_shape) {
  return st_reshape(tensor, new_ndim, new_shape);
}

/**
 * Concatenate multiple tensors along a given axis.
 *
 * Creates a new contiguous tensor with data from all input tensors
 * concatenated along the specified axis. All tensors must have the same
 * ndim and matching shapes on all axes except the concatenation axis.
 *
 * Example: concat two [3, 4] tensors along axis 0 -> [6, 4] tensor.
 *
 * @param tensors Array of input tensors (all must be non-NULL).
 * @param num_tensors Length of tensors array (>= 1).
 * @param axis Concatenation axis (0 <= axis < ndim).
 * @return New contiguous tensor (data copied from all inputs), or NULL on error.
 *         Caller must st_destroy() the returned tensor.
 */
FloatTensor *st_concat(const FloatTensor *const *tensors, size_t num_tensors,
                       size_t axis);

/**
 * Concatenate multiple tensors along a given axis (variadic).
 *
 * Convenience macro for st_concat() with a known tensor count at compile-time.
 * Automatically infers num_tensors from the array literal.
 *
 * Usage:
 *   FloatTensor *result = st_concat_varargs(t1, t2, t3, 0);
 *
 * NOTE: This is implemented as a macro and only works with array literals,
 *       not with pointer arrays.
 */
#define st_concat_varargs(tensors_array, axis) \
  st_concat((const FloatTensor *const *)&tensors_array, \
            sizeof(tensors_array) / sizeof((tensors_array)[0]), axis)
/**
 * Squeeze a tensor: remove all axes with size 1.
 *
 * Removes all dimensions of size 1 from the tensor shape.
 * For example, [2, 1, 3, 1, 4] becomes [2, 3, 4].
 *
 * @param tensor Input tensor.
 * @return New squeezed view (no data copy), or NULL on error.
 *         Caller must st_destroy() the returned tensor.
 */
FloatTensor *st_squeeze(const FloatTensor *tensor);

/**
 * Squeeze a single axis (if it has size 1).
 *
 * Removes a specific axis if its size is exactly 1.
 * If the axis size != 1, returns an error.
 *
 * @param tensor Input tensor.
 * @param axis Axis to squeeze (0 <= axis < ndim).
 * @return New squeezed view (no data copy), or NULL on error.
 *         Caller must st_destroy() the returned tensor.
 */
FloatTensor *st_squeeze_dim(const FloatTensor *tensor, size_t axis);

/**
 * Unsqueeze a tensor: insert a new axis of size 1.
 *
 * Inserts a new dimension of size 1 at the specified position.
 * For example, unsqueezing [2, 3, 4] at axis 1 gives [2, 1, 3, 4].
 *
 * @param tensor Input tensor.
 * @param axis Position to insert new axis (0 <= axis <= ndim).
 * @return New unsqueezed view (no data copy), or NULL on error.
 *         Caller must st_destroy() the returned tensor.
 */
FloatTensor *st_unsqueeze(const FloatTensor *tensor, size_t axis);

/**
 * Expand a tensor along a dimension by repeating.
 *
 * Repeats a tensor count times along the specified axis.
 * Size of axis must be 1 (broadcasting semantics).
 * For example, expanding [1, 3, 4] at axis 0 count=5 gives [5, 3, 4].
 *
 * @param tensor Input tensor (must have shape[axis] == 1).
 * @param axis Axis to expand.
 * @param count Number of repetitions (>= 1).
 * @return New expanded tensor (data copied), or NULL on error.
 *         Caller must st_destroy() the returned tensor.
 */
FloatTensor *st_expand(const FloatTensor *tensor, size_t axis, size_t count);

/**
 * Split a tensor along an axis into multiple sub-tensors.
 *
 * Splits a tensor into num_splits equal parts along the specified axis.
 * Allocates a new array of FloatTensor pointers (caller must free with free()).
 * Each sub-tensor is a view (no data copy).
 *
 * Example: split [12, 5] along axis 0 with 3 splits gives 3x [4, 5] tensors.
 *
 * @param tensor Input tensor.
 * @param axis Split axis.
 * @param num_splits Number of pieces to split into.
 * @param out_splits Output array of pointers to split tensors (must be freed by caller).
 * @return true on success, false on error (if not evenly divisible, etc.).
 *
 * NOTE: Caller must st_destroy() each tensor in out_splits and free(out_splits).
 */
bool st_split(const FloatTensor *tensor, size_t axis, size_t num_splits,
              FloatTensor ***out_splits);

#endif  // ST_SHAPE_OPS_H
