/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st.h"
#include "sm.h"

#include <log.h>
#include <stdlib.h>
#include <string.h>

static bool st_validate_shape(size_t ndim, const size_t *shape) {
  if (ndim == 0 || ndim > ST_MAX_DIMS || shape == NULL) {
    return false;
  }
  for (size_t i = 0; i < ndim; ++i) {
    if (shape[i] == 0) {
      return false;
    }
  }
  return true;
}

bool st_numel_from_shape(size_t ndim, const size_t *shape, size_t *out_numel) {
  if (!st_validate_shape(ndim, shape) || out_numel == NULL) {
    return false;
  }

  size_t numel = 1;
  for (size_t i = 0; i < ndim; ++i) {
    if (shape[i] > SIZE_MAX / numel) {
      return false;
    }
    numel *= shape[i];
  }

  *out_numel = numel;
  return true;
}

bool st_compute_default_strides(size_t ndim, const size_t *shape,
                                ptrdiff_t *out_strides) {
  if (!st_validate_shape(ndim, shape) || out_strides == NULL) {
    return false;
  }

  ptrdiff_t stride = 1;
  for (size_t i = ndim; i-- > 0;) {
    out_strides[i] = stride;
    if (shape[i] > (size_t)PTRDIFF_MAX / (size_t)stride) {
      return false;
    }
    stride *= (ptrdiff_t)shape[i];
  }

  return true;
}

FloatTensor *st_create(size_t ndim, const size_t *shape) {
  if (!st_validate_shape(ndim, shape)) {
    log_error("Error: st_create invalid shape.");
    return NULL;
  }

  size_t numel = 0;
  if (!st_numel_from_shape(ndim, shape, &numel)) {
    log_error("Error: st_create shape overflow.");
    return NULL;
  }

  FloatTensor *tensor = (FloatTensor *)calloc(1, sizeof(FloatTensor));
  if (!tensor) {
    log_error("Error: st_create tensor allocation failed.");
    return NULL;
  }

  tensor->values = (float *)calloc(numel, sizeof(float));
  if (!tensor->values) {
    log_error("Error: st_create data allocation failed.");
    free(tensor);
    return NULL;
  }

  tensor->ndim = ndim;
  memcpy(tensor->shape, shape, ndim * sizeof(size_t));
  if (!st_compute_default_strides(ndim, shape, tensor->strides)) {
    free(tensor->values);
    free(tensor);
    return NULL;
  }
  tensor->numel = numel;
  tensor->capacity = numel;
  tensor->owns_data = true;
  tensor->layout = ST_LAYOUT_CONTIGUOUS;

  return tensor;
}

FloatTensor *st_create_with_data(size_t ndim, const size_t *shape, float *data,
                                 size_t capacity, bool take_ownership) {
  if (!st_validate_shape(ndim, shape) || data == NULL) {
    log_error("Error: st_create_with_data invalid input.");
    return NULL;
  }

  size_t numel = 0;
  if (!st_numel_from_shape(ndim, shape, &numel) || numel > capacity) {
    log_error("Error: st_create_with_data invalid capacity.");
    return NULL;
  }

  FloatTensor *tensor = (FloatTensor *)calloc(1, sizeof(FloatTensor));
  if (!tensor) {
    log_error("Error: st_create_with_data tensor allocation failed.");
    return NULL;
  }

  tensor->ndim = ndim;
  memcpy(tensor->shape, shape, ndim * sizeof(size_t));
  if (!st_compute_default_strides(ndim, shape, tensor->strides)) {
    free(tensor);
    return NULL;
  }

  tensor->numel = numel;
  tensor->capacity = capacity;
  tensor->values = data;
  tensor->owns_data = take_ownership;
  tensor->layout = ST_LAYOUT_CONTIGUOUS;

  return tensor;
}

static bool st_offset_from_indices(const FloatTensor *tensor,
                                   const size_t *indices,
                                   size_t *out_offset) {
  if (tensor == NULL || indices == NULL || out_offset == NULL ||
      tensor->values == NULL || tensor->ndim == 0) {
    return false;
  }

  ptrdiff_t off = 0;
  for (size_t d = 0; d < tensor->ndim; ++d) {
    if (indices[d] >= tensor->shape[d]) {
      return false;
    }
    off += (ptrdiff_t)indices[d] * tensor->strides[d];
  }

  if (off < 0 || (size_t)off >= tensor->capacity) {
    return false;
  }

  *out_offset = (size_t)off;
  return true;
}

float st_get(const FloatTensor *tensor, const size_t *indices) {
  size_t off = 0;
  if (!st_offset_from_indices(tensor, indices, &off)) {
    log_error("Error: st_get index out of bounds or invalid tensor.");
    return 0.0f;
  }
  return tensor->values[off];
}

bool st_set(FloatTensor *tensor, const size_t *indices, float value) {
  size_t off = 0;
  if (!st_offset_from_indices(tensor, indices, &off)) {
    log_error("Error: st_set index out of bounds or invalid tensor.");
    return false;
  }
  tensor->values[off] = value;
  return true;
}

bool st_as_sm_view(const FloatTensor *tensor, FloatMatrix *out_view) {
  if (tensor == NULL || out_view == NULL || tensor->values == NULL) {
    return false;
  }
  if (tensor->ndim != 2 || !st_is_contiguous(tensor)) {
    return false;
  }

  out_view->rows = tensor->shape[0];
  out_view->cols = tensor->shape[1];
  out_view->capacity = tensor->numel;
  out_view->values = tensor->values;
  return true;
}

bool st_is_contiguous(const FloatTensor *tensor) {
  if (tensor == NULL || tensor->ndim == 0) {
    return false;
  }

  ptrdiff_t expected[ST_MAX_DIMS] = {0};
  if (!st_compute_default_strides(tensor->ndim, tensor->shape, expected)) {
    return false;
  }

  for (size_t i = 0; i < tensor->ndim; ++i) {
    if (tensor->strides[i] != expected[i]) {
      return false;
    }
  }
  return true;
}

bool st_reshape(FloatTensor *tensor, size_t new_ndim, const size_t *new_shape) {
  if (tensor == NULL || !st_is_contiguous(tensor) ||
      !st_validate_shape(new_ndim, new_shape)) {
    return false;
  }

  size_t new_numel = 0;
  if (!st_numel_from_shape(new_ndim, new_shape, &new_numel) ||
      new_numel != tensor->numel) {
    return false;
  }

  tensor->ndim = new_ndim;
  memset(tensor->shape, 0, sizeof(tensor->shape));
  memset(tensor->strides, 0, sizeof(tensor->strides));
  memcpy(tensor->shape, new_shape, new_ndim * sizeof(size_t));
  if (!st_compute_default_strides(new_ndim, new_shape, tensor->strides)) {
    return false;
  }

  tensor->layout = ST_LAYOUT_CONTIGUOUS;
  return true;
}

FloatTensor *st_view(FloatTensor *base, size_t ndim, const size_t *shape,
                     const ptrdiff_t *strides, size_t offset_elements) {
  if (base == NULL || base->values == NULL || !st_validate_shape(ndim, shape) ||
      strides == NULL || offset_elements >= base->capacity) {
    return NULL;
  }

  size_t numel = 0;
  if (!st_numel_from_shape(ndim, shape, &numel)) {
    return NULL;
  }

  FloatTensor *view = (FloatTensor *)calloc(1, sizeof(FloatTensor));
  if (!view) {
    return NULL;
  }

  view->ndim = ndim;
  memcpy(view->shape, shape, ndim * sizeof(size_t));
  memcpy(view->strides, strides, ndim * sizeof(ptrdiff_t));
  view->numel = numel;
  view->capacity = base->capacity - offset_elements;
  view->values = base->values + offset_elements;
  view->owns_data = false;
  view->layout = base->layout;

  return view;
}

FloatTensor *st_permute_view(FloatTensor *base, const size_t *perm) {
  if (base == NULL || perm == NULL || base->ndim == 0) {
    return NULL;
  }

  bool seen[ST_MAX_DIMS] = {false};
  size_t shape[ST_MAX_DIMS] = {0};
  ptrdiff_t strides[ST_MAX_DIMS] = {0};

  for (size_t i = 0; i < base->ndim; ++i) {
    if (perm[i] >= base->ndim || seen[perm[i]]) {
      return NULL;
    }
    seen[perm[i]] = true;
    shape[i] = base->shape[perm[i]];
    strides[i] = base->strides[perm[i]];
  }

  FloatTensor *view = st_view(base, base->ndim, shape, strides, 0);
  if (view) {
    view->layout = ST_LAYOUT_CONTIGUOUS;
  }
  return view;
}

FloatTensor *st_clone(const FloatTensor *src) {
  if (src == NULL || src->values == NULL || src->ndim == 0) {
    return NULL;
  }

  FloatTensor *dst = st_create(src->ndim, src->shape);
  if (!dst) {
    return NULL;
  }

  if (st_is_contiguous(src)) {
    memcpy(dst->values, src->values, src->numel * sizeof(float));
    dst->layout = src->layout;
    return dst;
  }

  size_t indices[ST_MAX_DIMS] = {0};
  for (size_t linear = 0; linear < src->numel; ++linear) {
    size_t tmp = linear;
    for (size_t d = src->ndim; d-- > 0;) {
      indices[d] = tmp % src->shape[d];
      tmp /= src->shape[d];
    }
    dst->values[linear] = st_get(src, indices);
  }

  dst->layout = src->layout;
  return dst;
}

void st_destroy(FloatTensor *tensor) {
  if (!tensor) {
    return;
  }

  if (tensor->owns_data && tensor->values) {
    free(tensor->values);
  }

  free(tensor);
}
