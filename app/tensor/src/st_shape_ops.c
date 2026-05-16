/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * Additional Shape Operation implementations for st_shape_ops.h
 * (squeeze, unsqueeze, expand, split)
 */

#include "st.h"

#include <stdlib.h>
#include <string.h>
FloatTensor *st_squeeze(const FloatTensor *tensor) {
  if (tensor == NULL || tensor->ndim == 0) {
    return NULL;
  }

  /* Count non-1 dimensions. */
  size_t new_ndim = 0;
  for (size_t i = 0; i < tensor->ndim; ++i) {
    if (tensor->shape[i] != 1) {
      new_ndim++;
    }
  }

  /* If all dimensions are 1 (scalar), squeezed shape is [1]. */
  if (new_ndim == 0) {
    new_ndim = 1;
  }

  /* Build squeezed shape and strides. */
  size_t new_shape[ST_MAX_DIMS];
  ptrdiff_t new_strides[ST_MAX_DIMS];
  size_t j = 0;
  for (size_t i = 0; i < tensor->ndim; ++i) {
    if (tensor->shape[i] != 1 || new_ndim == 1) {
      new_shape[j] = tensor->shape[i];
      new_strides[j] = tensor->strides[i];
      j++;
      if (new_ndim == 1) break;  /* Scalar case: keep first dim as 1. */
    }
  }

  return st_view((FloatTensor *)tensor, new_ndim, new_shape, new_strides, 0);
}

FloatTensor *st_squeeze_dim(const FloatTensor *tensor, size_t axis) {
  if (tensor == NULL || axis >= tensor->ndim ||
      tensor->shape[axis] != 1) {
    return NULL;
  }

  if (tensor->ndim == 1) {
    /* Cannot squeeze last dimension. */
    return NULL;
  }

  /* Build new shape/strides without axis. */
  size_t new_ndim = tensor->ndim - 1;
  size_t new_shape[ST_MAX_DIMS];
  ptrdiff_t new_strides[ST_MAX_DIMS];

  size_t j = 0;
  for (size_t i = 0; i < tensor->ndim; ++i) {
    if (i != axis) {
      new_shape[j] = tensor->shape[i];
      new_strides[j] = tensor->strides[i];
      j++;
    }
  }

  return st_view((FloatTensor *)tensor, new_ndim, new_shape, new_strides, 0);
}

FloatTensor *st_unsqueeze(const FloatTensor *tensor, size_t axis) {
  if (tensor == NULL || axis > tensor->ndim ||
      tensor->ndim + 1 > ST_MAX_DIMS) {
    return NULL;
  }

  /* Build new shape/strides with inserted dimension of size 1. */
  size_t new_ndim = tensor->ndim + 1;
  size_t new_shape[ST_MAX_DIMS];
  ptrdiff_t new_strides[ST_MAX_DIMS];

  for (size_t i = 0; i < axis; ++i) {
    new_shape[i] = tensor->shape[i];
    new_strides[i] = tensor->strides[i];
  }
  new_shape[axis] = 1;
  new_strides[axis] = 0;  /* Stride 0: dimension doesn't advance memory. */

  for (size_t i = axis + 1; i < new_ndim; ++i) {
    new_shape[i] = tensor->shape[i - 1];
    new_strides[i] = tensor->strides[i - 1];
  }

  return st_view((FloatTensor *)tensor, new_ndim, new_shape, new_strides, 0);
}

FloatTensor *st_expand(const FloatTensor *tensor, size_t axis, size_t count) {
  if (tensor == NULL || axis >= tensor->ndim ||
      tensor->shape[axis] != 1 || count == 0) {
    return NULL;
  }

  /* Create new contiguous tensor with expanded dimension. */
  size_t new_shape[ST_MAX_DIMS];
  memcpy(new_shape, tensor->shape, tensor->ndim * sizeof(size_t));
  new_shape[axis] = count;

  /* Allocate output. */
  FloatTensor *result = NULL;
  if (tensor->dtype == ST_DTYPE_BF16) {
    result = st_create_bf16(tensor->ndim, new_shape);
  } else {
    result = st_create(tensor->ndim, new_shape);
  }

  if (result == NULL) {
    return NULL;
  }

  /* Copy data count times along axis. */
  size_t src_numel = tensor->numel;
  for (size_t rep = 0; rep < count; ++rep) {
    for (size_t elem = 0; elem < src_numel; ++elem) {
      /* Compute multi-index in source. */
      size_t multi_idx[ST_MAX_DIMS];
      size_t temp = elem;
      for (int d = (int)tensor->ndim - 1; d >= 0; --d) {
        multi_idx[d] = temp % tensor->shape[d];
        temp /= tensor->shape[d];
      }

      /* Adjust axis index to replica. */
      multi_idx[axis] = rep;

      /* Compute flat index in result. */
      size_t result_idx = 0;
      size_t mult = 1;
      for (int d = (int)tensor->ndim - 1; d >= 0; --d) {
        result_idx += multi_idx[d] * mult;
        mult *= result->shape[d];
      }

      /* Copy element. */
      if (tensor->dtype == ST_DTYPE_BF16) {
        uint16_t bf16_val = ((uint16_t *)tensor->values)[elem];
        ((uint16_t *)result->values)[result_idx] = bf16_val;
      } else {
        result->values[result_idx] = tensor->values[elem];
      }
    }
  }

  return result;
}

bool st_split(const FloatTensor *tensor, size_t axis, size_t num_splits,
              FloatTensor ***out_splits) {
  if (tensor == NULL || axis >= tensor->ndim || num_splits == 0 ||
      out_splits == NULL) {
    return false;
  }

  /* Check if axis is evenly divisible. */
  if (tensor->shape[axis] % num_splits != 0) {
    return false;
  }

  size_t split_size = tensor->shape[axis] / num_splits;

  /* Allocate output array. */
  *out_splits = (FloatTensor **)malloc(num_splits * sizeof(FloatTensor *));
  if (*out_splits == NULL) {
    return false;
  }

  /* Create views for each split. */
  for (size_t s = 0; s < num_splits; ++s) {
    size_t new_shape[ST_MAX_DIMS];
    ptrdiff_t new_strides[ST_MAX_DIMS];
    memcpy(new_shape, tensor->shape, tensor->ndim * sizeof(size_t));
    memcpy(new_strides, tensor->strides, tensor->ndim * sizeof(ptrdiff_t));
    new_shape[axis] = split_size;

    /* Compute element offset for this split. */
    size_t offset_elements = 0;
    ptrdiff_t stride = tensor->strides[axis];
    if (stride > 0) {
      offset_elements = (s * split_size * stride) / (ptrdiff_t)sizeof(float);
    } else if (stride < 0) {
      offset_elements = (s * split_size * (-stride)) / (ptrdiff_t)sizeof(float);
    }

    (*out_splits)[s] = st_view((FloatTensor *)tensor, tensor->ndim, new_shape,
                               new_strides, offset_elements);
    if ((*out_splits)[s] == NULL) {
      /* Cleanup on error. */
      for (size_t i = 0; i < s; ++i) {
        st_destroy((*out_splits)[i]);
      }
      free(*out_splits);
      *out_splits = NULL;
      return false;
    }
  }

  return true;
}
