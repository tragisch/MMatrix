/**
 * @file st_shape_ops.h
 * @brief Public API for tensor shape/view transformation operations.
 */

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

/**
 * @brief Flatten a range of axes into a single axis.
 * @param tensor Input tensor.
 * @param start_axis First axis to merge (inclusive).
 * @param end_axis Last axis to merge (exclusive).
 * @return New flattened view (no data copy), or NULL on error.
 */
FloatTensor *st_flatten(const FloatTensor *tensor, size_t start_axis,
                        size_t end_axis);

/**
 * @brief Flatten all axes into a 1D view.
 * @param tensor Input tensor.
 * @return New 1D view (no data copy), or NULL on error.
 */
FloatTensor *st_flatten_all(const FloatTensor *tensor);

/**
 * @brief Permute tensor axes.
 * @param tensor Input tensor.
 * @param axes Permutation array where `axes[i]` maps output axis `i`.
 * @return New permuted view (no data copy), or NULL on error.
 */
FloatTensor *st_permute(const FloatTensor *tensor, const size_t *axes);

/**
 * @brief Convenience alias for @ref st_reshape.
 */
static inline bool st_reshape_to(FloatTensor *tensor, size_t new_ndim,
                                 const size_t *new_shape) {
  return st_reshape(tensor, new_ndim, new_shape);
}

/**
 * @brief Concatenate multiple tensors along one axis.
 * @param tensors Array of input tensors.
 * @param num_tensors Number of entries in @p tensors.
 * @param axis Concatenation axis.
 * @return New contiguous tensor on success, or NULL on error.
 */
FloatTensor *st_concat(const FloatTensor *const *tensors, size_t num_tensors,
                       size_t axis);

/**
 * @brief Convenience macro around @ref st_concat for array literals.
 */
#define st_concat_varargs(tensors_array, axis) \
  st_concat((const FloatTensor *const *)&tensors_array, \
            sizeof(tensors_array) / sizeof((tensors_array)[0]), axis)

/**
 * @brief Remove all dimensions of size 1.
 * @param tensor Input tensor.
 * @return New squeezed view (no data copy), or NULL on error.
 */
FloatTensor *st_squeeze(const FloatTensor *tensor);

/**
 * @brief Remove one dimension if and only if its size is 1.
 * @param tensor Input tensor.
 * @param axis Axis to remove.
 * @return New squeezed view (no data copy), or NULL on error.
 */
FloatTensor *st_squeeze_dim(const FloatTensor *tensor, size_t axis);

/**
 * @brief Insert a new dimension of size 1.
 * @param tensor Input tensor.
 * @param axis Insert position in `[0, ndim]`.
 * @return New unsqueezed view (no data copy), or NULL on error.
 */
FloatTensor *st_unsqueeze(const FloatTensor *tensor, size_t axis);

/**
 * @brief Expand a size-1 dimension by repetition.
 * @param tensor Input tensor.
 * @param axis Axis to expand (must currently be size 1).
 * @param count Replication count.
 * @return New tensor with copied data, or NULL on error.
 */
FloatTensor *st_expand(const FloatTensor *tensor, size_t axis, size_t count);

/**
 * @brief Split a tensor into equal parts along one axis.
 * @param tensor Input tensor.
 * @param axis Split axis.
 * @param num_splits Number of equally sized splits.
 * @param out_splits Output array of split tensor handles.
 * @retval true Success.
 * @retval false Invalid input or uneven split.
 *
 * @note Caller must destroy each split tensor and then free `*out_splits`.
 */
bool st_split(const FloatTensor *tensor, size_t axis, size_t num_splits,
              FloatTensor ***out_splits);

#endif  // ST_SHAPE_OPS_H
