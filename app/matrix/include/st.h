/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_H
#define ST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ST_MAX_DIMS 8

typedef enum StLayout {
  ST_LAYOUT_CONTIGUOUS = 0,
  ST_LAYOUT_NCHW = 1,
  ST_LAYOUT_NHWC = 2,
  ST_LAYOUT_TBF = 3,  // [time, batch, feature]
  ST_LAYOUT_BTF = 4,  // [batch, time, feature]
} StLayout;

typedef struct FloatMatrix FloatMatrix;

typedef struct FloatTensor {
  size_t ndim;
  size_t shape[ST_MAX_DIMS];
  ptrdiff_t strides[ST_MAX_DIMS];
  size_t numel;
  size_t capacity;
  float *values;
  bool owns_data;
  StLayout layout;
} FloatTensor;

/**
 * @brief Computes row-major contiguous strides for a shape.
 *
 * @param ndim Number of dimensions.
 * @param shape Shape array with ndim entries.
 * @param out_strides Output array with ndim entries.
 *
 * @return true if computed, false on invalid input or overflow.
 */
bool st_compute_default_strides(size_t ndim, const size_t *shape,
                                ptrdiff_t *out_strides);

/**
 * @brief Computes number of elements in a tensor shape.
 *
 * @param ndim Number of dimensions.
 * @param shape Shape array with ndim entries.
 * @param out_numel Output number of elements.
 *
 * @return true on success, false on overflow or invalid input.
 */
bool st_numel_from_shape(size_t ndim, const size_t *shape, size_t *out_numel);

/**
 * @brief Creates a new contiguous float tensor and zero-initializes memory.
 */
FloatTensor *st_create(size_t ndim, const size_t *shape);

/**
 * @brief Creates tensor wrapper around pre-existing data pointer.
 *
 * Caller can decide ownership via take_ownership. No data copy is performed.
 */
FloatTensor *st_create_with_data(size_t ndim, const size_t *shape, float *data,
                                 size_t capacity, bool take_ownership);

/**
 * @brief Creates deep copy of tensor data into contiguous storage.
 */
FloatTensor *st_clone(const FloatTensor *src);

/**
 * @brief Creates a view on base tensor data with given shape/strides and offset.
 *
 * NOTE: returned tensor does not own underlying data.
 */
FloatTensor *st_view(FloatTensor *base, size_t ndim, const size_t *shape,
                     const ptrdiff_t *strides, size_t offset_elements);

/**
 * @brief Returns true if tensor uses standard row-major contiguous strides.
 */
bool st_is_contiguous(const FloatTensor *tensor);

/**
 * @brief Reshapes tensor in-place (only if contiguous and numel unchanged).
 */
bool st_reshape(FloatTensor *tensor, size_t new_ndim, const size_t *new_shape);

/**
 * @brief Creates a permuted view of a tensor.
 *
 * @param base Source tensor.
 * @param perm Permutation array of length base->ndim.
 */
FloatTensor *st_permute_view(FloatTensor *base, const size_t *perm);

/**
 * @brief Gets scalar value at given multi-index.
 */
float st_get(const FloatTensor *tensor, const size_t *indices);

/**
 * @brief Sets scalar value at given multi-index.
 *
 * @return true on success, false on out-of-bounds or invalid inputs.
 */
bool st_set(FloatTensor *tensor, const size_t *indices, float value);

/**
 * @brief Exposes a 2D contiguous tensor as FloatMatrix view without copy.
 *
 * The resulting view points directly to tensor memory and does not own data.
 * Caller must NOT free tensor data via sm_destroy on the view.
 *
 * @param tensor Input tensor (must be 2D and contiguous).
 * @param out_view Output matrix view metadata.
 *
 * @return true on success, false if shape/layout is incompatible.
 */
bool st_as_sm_view(const FloatTensor *tensor, FloatMatrix *out_view);

/**
 * @brief Releases tensor object and owned data if owns_data==true.
 */
void st_destroy(FloatTensor *tensor);

#endif  // ST_H
