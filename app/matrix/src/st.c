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
#include <math.h>
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

/* ---- Element-wise in-place operations ---- */

bool st_inplace_add(FloatTensor *a, const FloatTensor *b) {
  if (a == NULL || a->values == NULL) {
    return false;
  }
  if (b == NULL) {
    return true; /* no-op */
  }
  if (b->values == NULL || a->numel != b->numel) {
    log_error("Error: st_inplace_add size mismatch.");
    return false;
  }
  if (!st_is_contiguous(a) || !st_is_contiguous(b)) {
    log_error("Error: st_inplace_add requires contiguous tensors.");
    return false;
  }

  for (size_t i = 0; i < a->numel; ++i) {
    a->values[i] += b->values[i];
  }
  return true;
}

bool st_inplace_sub(FloatTensor *a, const FloatTensor *b) {
  if (a == NULL || a->values == NULL || b == NULL || b->values == NULL) {
    return false;
  }
  if (a->numel != b->numel) {
    log_error("Error: st_inplace_sub size mismatch.");
    return false;
  }
  if (!st_is_contiguous(a) || !st_is_contiguous(b)) {
    log_error("Error: st_inplace_sub requires contiguous tensors.");
    return false;
  }

  for (size_t i = 0; i < a->numel; ++i) {
    a->values[i] -= b->values[i];
  }
  return true;
}

bool st_inplace_scale(FloatTensor *t, float scalar) {
  if (t == NULL || t->values == NULL) {
    return false;
  }
  if (!st_is_contiguous(t)) {
    log_error("Error: st_inplace_scale requires contiguous tensor.");
    return false;
  }

  for (size_t i = 0; i < t->numel; ++i) {
    t->values[i] *= scalar;
  }
  return true;
}

bool st_inplace_elementwise_multiply(FloatTensor *a, const FloatTensor *b) {
  if (a == NULL || a->values == NULL || b == NULL || b->values == NULL) {
    return false;
  }
  if (a->numel != b->numel) {
    log_error("Error: st_inplace_elementwise_multiply size mismatch.");
    return false;
  }
  if (!st_is_contiguous(a) || !st_is_contiguous(b)) {
    log_error(
        "Error: st_inplace_elementwise_multiply requires contiguous tensors.");
    return false;
  }

  for (size_t i = 0; i < a->numel; ++i) {
    a->values[i] *= b->values[i];
  }
  return true;
}

bool st_fill(FloatTensor *t, float value) {
  if (t == NULL || t->values == NULL) {
    return false;
  }
  if (!st_is_contiguous(t)) {
    log_error("Error: st_fill requires contiguous tensor.");
    return false;
  }

  if (value == 0.0f) {
    memset(t->values, 0, t->numel * sizeof(float));
  } else {
    for (size_t i = 0; i < t->numel; ++i) {
      t->values[i] = value;
    }
  }
  return true;
}

/* ---- Activation functions on tensors ---- */

bool st_apply_relu(FloatTensor *t) {
  if (t == NULL || t->values == NULL) {
    return false;
  }
  if (!st_is_contiguous(t)) {
    log_error("Error: st_apply_relu requires contiguous tensor.");
    return false;
  }

  for (size_t i = 0; i < t->numel; ++i) {
    if (t->values[i] < 0.0f) {
      t->values[i] = 0.0f;
    }
  }
  return true;
}

bool st_apply_relu_backward(const FloatTensor *activation, FloatTensor *grad) {
  if (activation == NULL || activation->values == NULL || grad == NULL ||
      grad->values == NULL) {
    return false;
  }
  if (activation->numel != grad->numel) {
    log_error("Error: st_apply_relu_backward size mismatch.");
    return false;
  }
  if (!st_is_contiguous(activation) || !st_is_contiguous(grad)) {
    log_error(
        "Error: st_apply_relu_backward requires contiguous tensors.");
    return false;
  }

  for (size_t i = 0; i < grad->numel; ++i) {
    if (activation->values[i] <= 0.0f) {
      grad->values[i] = 0.0f;
    }
  }
  return true;
}

/* ---- Reduction: sum over axes ---- */

FloatTensor *st_sum_axes(const FloatTensor *t, const size_t *axes,
                         size_t num_axes) {
  if (t == NULL || t->values == NULL || axes == NULL || num_axes == 0) {
    return NULL;
  }
  if (!st_is_contiguous(t)) {
    log_error("Error: st_sum_axes requires contiguous tensor.");
    return NULL;
  }
  if (num_axes > t->ndim) {
    log_error("Error: st_sum_axes more axes than dimensions.");
    return NULL;
  }

  /* Validate axes and build a reduction mask. */
  bool reduce[ST_MAX_DIMS] = {false};
  for (size_t i = 0; i < num_axes; ++i) {
    if (axes[i] >= t->ndim) {
      log_error("Error: st_sum_axes axis out of range.");
      return NULL;
    }
    if (reduce[axes[i]]) {
      log_error("Error: st_sum_axes duplicate axis.");
      return NULL;
    }
    reduce[axes[i]] = true;
  }

  /* Compute output shape: keep non-reduced dimensions. */
  size_t out_shape[ST_MAX_DIMS] = {0};
  size_t out_ndim = 0;
  for (size_t d = 0; d < t->ndim; ++d) {
    if (!reduce[d]) {
      out_shape[out_ndim++] = t->shape[d];
    }
  }

  /* Edge case: all dims reduced → scalar tensor with ndim=1, shape={1}. */
  if (out_ndim == 0) {
    out_ndim = 1;
    out_shape[0] = 1;
  }

  FloatTensor *result = st_create(out_ndim, out_shape);
  if (!result) {
    return NULL;
  }

  /* Generic N-dimensional summation via multi-index iteration. */
  size_t indices[ST_MAX_DIMS] = {0};

  for (size_t linear = 0; linear < t->numel; ++linear) {
    /* Decompose linear index into multi-index. */
    size_t tmp = linear;
    for (size_t d = t->ndim; d-- > 0;) {
      indices[d] = tmp % t->shape[d];
      tmp /= t->shape[d];
    }

    /* Compute output linear index from non-reduced dimensions. */
    size_t out_linear = 0;
    size_t out_stride = 1;
    size_t out_d = out_ndim;
    for (size_t d = t->ndim; d-- > 0;) {
      if (!reduce[d]) {
        --out_d;
        out_linear += indices[d] * out_stride;
        out_stride *= out_shape[out_d];
      }
    }

    result->values[out_linear] += t->values[linear];
  }

  return result;
}

/* ---- Padding ---- */

FloatTensor *st_pad_nchw(const FloatTensor *input, size_t pad_h, size_t pad_w,
                         float value) {
  if (input == NULL || input->values == NULL || input->ndim != 4) {
    log_error("Error: st_pad_nchw expects valid 4D tensor.");
    return NULL;
  }
  if (!st_is_contiguous(input)) {
    log_error("Error: st_pad_nchw requires contiguous tensor.");
    return NULL;
  }
  if (pad_h == 0 && pad_w == 0) {
    return st_clone(input);
  }

  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];

  const size_t new_h = h + 2 * pad_h;
  const size_t new_w = w + 2 * pad_w;

  size_t out_shape[4] = {n, c, new_h, new_w};
  FloatTensor *result = st_create(4, out_shape);
  if (!result) {
    return NULL;
  }

  /* Fill with padding value if non-zero (st_create already zeroed). */
  if (value != 0.0f) {
    for (size_t i = 0; i < result->numel; ++i) {
      result->values[i] = value;
    }
  }

  /* Copy input data into the padded region. */
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      for (size_t hi = 0; hi < h; ++hi) {
        const float *src =
            input->values + ((ni * c + ci) * h + hi) * w;
        float *dst = result->values +
                     ((ni * c + ci) * new_h + (hi + pad_h)) * new_w + pad_w;
        memcpy(dst, src, w * sizeof(float));
      }
    }
  }

  return result;
}
