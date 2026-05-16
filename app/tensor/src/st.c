/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_CLEANUP_FREE
#define ST_CLEANUP_FREE(ptr) do { if (ptr) { free(ptr); ptr = NULL; } } while (0)
#endif

/* ---- Macros for bf16 promotion in element-wise operations ---- */

/* Binary op bf16 promotion: promotes bf16 args to f32, recurses, converts result back. */
#define ST_INPLACE_BINARY_BF16_PROMOTE(A, B, FUNC_NAME) \
  do { \
    if ((A)->dtype == ST_DTYPE_BF16 || (B)->dtype == ST_DTYPE_BF16) { \
      FloatTensor *a_f32 = st_to_f32(A); \
      FloatTensor *b_f32 = ((B)->dtype == ST_DTYPE_BF16) ? st_to_f32(B) : (FloatTensor *)(B); \
      if (!a_f32 || !b_f32) { \
        st_destroy(a_f32); \
        if (b_f32 != (B)) st_destroy(b_f32); \
        return false; \
      } \
      bool ok = FUNC_NAME(a_f32, b_f32); \
      if (ok && (A)->dtype == ST_DTYPE_BF16) { \
        st_f32_to_bf16_bulk(a_f32->values, (uint16_t *)(A)->values, (A)->numel); \
      } else if (ok) { \
        memcpy((A)->values, a_f32->values, (A)->numel * sizeof(float)); \
      } \
      st_destroy(a_f32); \
      if (b_f32 != (B)) st_destroy(b_f32); \
      return ok; \
    } \
  } while (0)

/* Unary op bf16 promotion: promotes bf16 arg to f32, recurses, converts result back. */
#define ST_INPLACE_UNARY_BF16_PROMOTE(T, FUNC_NAME) \
  do { \
    if ((T)->dtype == ST_DTYPE_BF16) { \
      FloatTensor *t_f32 = st_to_f32(T); \
      if (!t_f32) return false; \
      bool ok = FUNC_NAME(t_f32); \
      if (ok) { \
        st_f32_to_bf16_bulk(t_f32->values, (uint16_t *)(T)->values, (T)->numel); \
      } \
      st_destroy(t_f32); \
      return ok; \
    } \
  } while (0)


#include "st.h"
#include "st_buffer.h"
#include "st_dtype.h"
#include "st_gpu_guard.h"
#include "sm.h"
#include "st_bf16_utils.h"

#include <stdio.h>
#ifndef ST_LOG_ERROR
#define ST_LOG_ERROR(fmt, ...) fprintf(stderr, "[st][error] " fmt "\n", ##__VA_ARGS__)
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(USE_ACCELERATE)
#define BLASINT int
#include <Accelerate/Accelerate.h>
#endif

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

static bool st_add_size_checked(size_t lhs, size_t rhs, size_t *out_sum) {
  if (out_sum == NULL || lhs > SIZE_MAX - rhs) {
    return false;
  }
  *out_sum = lhs + rhs;
  return true;
}

static bool st_view_is_in_bounds(const FloatTensor *base, size_t ndim,
                                 const size_t *shape,
                                 const ptrdiff_t *strides,
                                 size_t offset_elements) {
  if (base == NULL || shape == NULL || strides == NULL) {
    return false;
  }

  /* Effective capacity in logical elements (bf16 packs 2 per float slot). */
  size_t effective_cap = base->capacity;
  if (base->dtype == ST_DTYPE_BF16 && base->buf) {
    effective_cap = base->buf->size_bytes / st_dtype_size(ST_DTYPE_BF16);
  }

  if (offset_elements >= effective_cap) {
    return false;
  }

  size_t max_offset = 0;
  for (size_t i = 0; i < ndim; ++i) {
    if (strides[i] < 0) {
      return false;
    }

    if (shape[i] <= 1 || strides[i] == 0) {
      continue;
    }

    const size_t extent = shape[i] - 1;
    const size_t stride = (size_t)strides[i];
    if (extent > SIZE_MAX / stride) {
      return false;
    }

    const size_t dim_max = extent * stride;
    if (!st_add_size_checked(max_offset, dim_max, &max_offset)) {
      return false;
    }
  }

  size_t view_end = 0;
  if (!st_add_size_checked(offset_elements, max_offset, &view_end)) {
    return false;
  }

  return view_end < effective_cap;
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
    ST_LOG_ERROR("st_create invalid shape.");
    return NULL;
  }

  size_t numel = 0;
  if (!st_numel_from_shape(ndim, shape, &numel)) {
    ST_LOG_ERROR("st_create shape overflow.");
    return NULL;
  }

  StBuffer *buf = st_buffer_alloc(numel);
  if (!buf) {
    ST_LOG_ERROR("st_create buffer allocation failed.");
    return NULL;
  }

  FloatTensor *tensor = (FloatTensor *)calloc(1, sizeof(FloatTensor));
  if (!tensor) {
    ST_LOG_ERROR("st_create tensor allocation failed.");
    st_buffer_release(buf);
    return NULL;
  }

  bool ok = true;
  tensor->ndim = ndim;
  memcpy(tensor->shape, shape, ndim * sizeof(size_t));
  if (!st_compute_default_strides(ndim, shape, tensor->strides)) {
    ok = false;
    goto cleanup;
  }
  tensor->numel = numel;
  tensor->layout = ST_LAYOUT_CONTIGUOUS;
  tensor->dtype = ST_DTYPE_F32;

  /* Buffer-backed storage */
  tensor->buf = buf;
  tensor->view_offset = 0;
  tensor->view_src = NULL;
  tensor->extra = NULL;

  /* Derived aliases (backward compat) */
  tensor->values = buf->data;
  tensor->capacity = buf->capacity;
  tensor->owns_data = true;

  return tensor;

cleanup:
  if (tensor) {
    st_buffer_release(buf);
    ST_CLEANUP_FREE(tensor);
  }
  return NULL;
}

FloatTensor *st_create_with_data(size_t ndim, const size_t *shape, float *data,
                                 size_t capacity, bool take_ownership) {
  if (!st_validate_shape(ndim, shape) || data == NULL) {
    ST_LOG_ERROR("st_create_with_data invalid input.");
    return NULL;
  }

  size_t numel = 0;
  if (!st_numel_from_shape(ndim, shape, &numel) || numel > capacity) {
    ST_LOG_ERROR("st_create_with_data invalid capacity.");
    return NULL;
  }

  StBuffer *buf = st_buffer_from_ptr(data, capacity, take_ownership);
  if (!buf) {
    ST_LOG_ERROR("st_create_with_data buffer creation failed.");
    return NULL;
  }

  FloatTensor *tensor = (FloatTensor *)calloc(1, sizeof(FloatTensor));
  if (!tensor) {
    ST_LOG_ERROR("st_create_with_data tensor allocation failed.");
    st_buffer_release(buf);
    return NULL;
  }

  bool ok = true;
  tensor->ndim = ndim;
  memcpy(tensor->shape, shape, ndim * sizeof(size_t));
  if (!st_compute_default_strides(ndim, shape, tensor->strides)) {
    ok = false;
    goto cleanup;
  }

  tensor->numel = numel;
  tensor->layout = ST_LAYOUT_CONTIGUOUS;
  tensor->dtype = ST_DTYPE_F32;

  /* Buffer-backed storage */
  tensor->buf = buf;
  tensor->view_offset = 0;
  tensor->view_src = NULL;
  tensor->extra = NULL;

  /* Derived aliases (backward compat) */
  tensor->values = buf->data;
  tensor->capacity = buf->capacity;
  tensor->owns_data = take_ownership;

  return tensor;

cleanup:
  if (tensor) {
    st_buffer_release(buf);
    ST_CLEANUP_FREE(tensor);
  }
  return NULL;
}

FloatTensor *st_create_bf16(size_t ndim, const size_t *shape) {
  if (!st_validate_shape(ndim, shape)) {
    ST_LOG_ERROR("st_create_bf16 invalid shape.");
    return NULL;
  }

  size_t numel = 0;
  if (!st_numel_from_shape(ndim, shape, &numel)) {
    ST_LOG_ERROR("st_create_bf16 shape overflow.");
    return NULL;
  }

  const size_t alloc_bytes = numel * st_dtype_size(ST_DTYPE_BF16);
  if (alloc_bytes / st_dtype_size(ST_DTYPE_BF16) != numel) {
    ST_LOG_ERROR("st_create_bf16 byte overflow.");
    return NULL;
  }

  StBuffer *buf = st_buffer_alloc_bytes(alloc_bytes);
  if (!buf) {
    ST_LOG_ERROR("st_create_bf16 buffer allocation failed.");
    return NULL;
  }

  FloatTensor *tensor = (FloatTensor *)calloc(1, sizeof(FloatTensor));
  if (!tensor) {
    ST_LOG_ERROR("st_create_bf16 tensor allocation failed.");
    st_buffer_release(buf);
    return NULL;
  }

  bool ok = true;
  tensor->ndim = ndim;
  memcpy(tensor->shape, shape, ndim * sizeof(size_t));
  if (!st_compute_default_strides(ndim, shape, tensor->strides)) {
    ok = false;
    goto cleanup;
  }
  tensor->numel = numel;
  tensor->layout = ST_LAYOUT_CONTIGUOUS;
  tensor->dtype = ST_DTYPE_BF16;

  /* Buffer-backed storage */
  tensor->buf = buf;
  tensor->view_offset = 0;
  tensor->view_src = NULL;
  tensor->extra = NULL;

  /* Raw data pointer (NOT valid float data — use bf16 accessors). */
  tensor->values = buf->data;
  tensor->capacity = numel;
  tensor->owns_data = true;

  return tensor;

cleanup:
  if (tensor) {
    st_buffer_release(buf);
    ST_CLEANUP_FREE(tensor);
  }
  return NULL;
}

FloatTensor *st_to_f32(const FloatTensor *src) {
  if (src == NULL || src->values == NULL || src->ndim == 0) {
    return NULL;
  }
  ST_ASSERT_NOT_PENDING(src);

  if (src->dtype == ST_DTYPE_F32) {
    return st_clone(src);
  }

  /* bf16 → f32 */
  FloatTensor *dst = st_create(src->ndim, src->shape);
  if (!dst) {
    return NULL;
  }

  if (st_is_contiguous(src)) {
    const uint16_t *bf_data = (const uint16_t *)src->values;
    st_bf16_to_f32_bulk(bf_data, dst->values, src->numel);
  } else {
    size_t indices[ST_MAX_DIMS] = {0};
    for (size_t linear = 0; linear < src->numel; ++linear) {
      size_t tmp = linear;
      for (size_t d = src->ndim; d-- > 0;) {
        indices[d] = tmp % src->shape[d];
        tmp /= src->shape[d];
      }
      dst->values[linear] = st_get(src, indices);
    }
  }

  dst->layout = src->layout;
  return dst;
}

FloatTensor *st_to_bf16(const FloatTensor *src) {
  if (src == NULL || src->values == NULL || src->ndim == 0) {
    return NULL;
  }
  ST_ASSERT_NOT_PENDING(src);

  if (src->dtype == ST_DTYPE_BF16) {
    return st_clone(src);
  }

  /* f32 → bf16 */
  FloatTensor *dst = st_create_bf16(src->ndim, src->shape);
  if (!dst) {
    return NULL;
  }

  if (st_is_contiguous(src)) {
    uint16_t *bf_data = (uint16_t *)dst->values;
    st_f32_to_bf16_bulk(src->values, bf_data, src->numel);
  } else {
    size_t indices[ST_MAX_DIMS] = {0};
    for (size_t linear = 0; linear < src->numel; ++linear) {
      size_t tmp = linear;
      for (size_t d = src->ndim; d-- > 0;) {
        indices[d] = tmp % src->shape[d];
        tmp /= src->shape[d];
      }
      if (!st_set(dst, indices, st_get(src, indices))) {
        st_destroy(dst);
        return NULL;
      }
    }
  }

  dst->layout = src->layout;
  return dst;
}

static bool st_offset_from_indices(const FloatTensor *tensor,
                                   const size_t *indices,
                                   size_t *out_offset) {
  if (tensor == NULL || indices == NULL || out_offset == NULL ||
      tensor->values == NULL || tensor->ndim == 0) {
    return false;
  }

  size_t off = 0;
  for (size_t d = 0; d < tensor->ndim; ++d) {
    if (indices[d] >= tensor->shape[d]) {
      return false;
    }

    if (tensor->strides[d] < 0) {
      return false;
    }

    const size_t stride = (size_t)tensor->strides[d];
    if (indices[d] == 0 || stride == 0) {
      continue;
    }
    if (indices[d] > SIZE_MAX / stride) {
      return false;
    }

    const size_t dim_offset = indices[d] * stride;
    if (dim_offset > SIZE_MAX - off) {
      return false;
    }
    off += dim_offset;
  }

  if (off >= tensor->capacity) {
    /* For bf16 views, capacity is already in logical element units.
     * This check remains valid for both f32 and bf16. */
    return false;
  }

  *out_offset = off;
  return true;
}

float st_get(const FloatTensor *tensor, const size_t *indices) {
  size_t off = 0;
  if (!st_offset_from_indices(tensor, indices, &off)) {
    ST_LOG_ERROR("st_get index out of bounds or invalid tensor.");
    return 0.0f;
  }
  ST_ASSERT_NOT_PENDING(tensor);
  if (tensor->dtype == ST_DTYPE_BF16) {
    const uint16_t *bp = (const uint16_t *)tensor->values;
    return st_bf16_to_f32(bp[off]);
  }
  return tensor->values[off];
}

bool st_set(FloatTensor *tensor, const size_t *indices, float value) {
  size_t off = 0;
  if (!st_offset_from_indices(tensor, indices, &off)) {
    ST_LOG_ERROR("st_set index out of bounds or invalid tensor.");
    return false;
  }
  ST_ASSERT_NOT_PENDING(tensor);
  if (tensor->dtype == ST_DTYPE_BF16) {
    uint16_t *bp = (uint16_t *)tensor->values;
    bp[off] = st_f32_to_bf16(value);
    return true;
  }
  tensor->values[off] = value;
  return true;
}

bool st_as_sm_view(const FloatTensor *tensor, FloatMatrix *out_view) {
  if (tensor == NULL || out_view == NULL || tensor->values == NULL) {
    return false;
  }
  if (tensor->dtype != ST_DTYPE_F32) {
    ST_LOG_ERROR("st_as_sm_view requires f32 tensor. Use st_to_f32() to convert bf16 tensors first.");
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
      strides == NULL ||
      !st_view_is_in_bounds(base, ndim, shape, strides, offset_elements)) {
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
  view->layout = base->layout;

  /* Buffer-backed storage: share base's buffer */
  view->buf = st_buffer_retain(base->buf);
  view->view_offset = base->view_offset + offset_elements;
  view->view_src = base;
  view->dtype = base->dtype;
  view->extra = NULL;
  view->extra_free = NULL;

  /* Derived aliases (backward compat).
   * For bf16 tensors the buffer stores uint16_t elements packed into the
   * float* allocation.  offset_elements counts logical elements, so
   * we must convert to a byte offset when the element size differs.  */
  if (base->dtype == ST_DTYPE_BF16) {
    view->values = (float *)((char *)view->buf->data +
                             view->view_offset * st_dtype_size(ST_DTYPE_BF16));
    /* capacity in logical bf16 elements */
    const size_t total_bf16_elems =
        view->buf->size_bytes / st_dtype_size(ST_DTYPE_BF16);
    view->capacity =
        (total_bf16_elems > view->view_offset)
            ? total_bf16_elems - view->view_offset
            : 0;
  } else {
    view->values = view->buf->data + view->view_offset;
    view->capacity = view->buf->capacity - view->view_offset;
  }
  view->owns_data = false;

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
    view->layout = base->layout;
  }
  return view;
}

FloatTensor *st_clone(const FloatTensor *src) {
  if (src == NULL || src->values == NULL || src->ndim == 0) {
    return NULL;
  }
  ST_ASSERT_NOT_PENDING(src);

  /* bf16 clone: byte-copy of the packed storage. */
  if (src->dtype == ST_DTYPE_BF16) {
    FloatTensor *dst = st_create_bf16(src->ndim, src->shape);
    if (!dst) {
      return NULL;
    }
    if (st_is_contiguous(src)) {
      memcpy(dst->values, src->values,
             src->numel * st_dtype_size(ST_DTYPE_BF16));
    } else {
      /* Non-contiguous bf16: element-wise via scalar conversion. */
      size_t indices[ST_MAX_DIMS] = {0};
      for (size_t linear = 0; linear < src->numel; ++linear) {
        size_t tmp = linear;
        for (size_t d = src->ndim; d-- > 0;) {
          indices[d] = tmp % src->shape[d];
          tmp /= src->shape[d];
        }
        float v = st_get(src, indices);
        st_set(dst, indices, v);
      }
    }
    dst->layout = src->layout;
    return dst;
  }

  /* f32 clone (original path). */
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

  if (tensor->extra && tensor->extra_free) {
    tensor->extra_free(tensor->extra);
    tensor->extra = NULL;
  }

  if (tensor->buf) {
    st_buffer_release(tensor->buf);
    tensor->buf = NULL;
  }

  tensor->values = NULL;
  free(tensor);
}

void st_tensor_sync(FloatTensor *tensor) {
  if (tensor && tensor->buf) {
    st_buffer_wait_gpu(tensor->buf);
  }
}

size_t st_tensor_ndim(const FloatTensor *tensor) {
  return tensor ? tensor->ndim : 0u;
}

const size_t *st_tensor_shape(const FloatTensor *tensor) {
  return tensor ? tensor->shape : NULL;
}

size_t st_tensor_numel(const FloatTensor *tensor) {
  return tensor ? tensor->numel : 0u;
}

StDtype st_tensor_dtype(const FloatTensor *tensor) {
  return tensor ? tensor->dtype : ST_DTYPE_F32;
}

const float *st_tensor_data(const FloatTensor *tensor) {
  return tensor ? tensor->values : NULL;
}

float *st_tensor_mutable_data(FloatTensor *tensor) {
  return tensor ? tensor->values : NULL;
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
    ST_LOG_ERROR("st_inplace_add size mismatch.");
    return false;
  }
  if (!st_is_contiguous(a) || !st_is_contiguous(b)) {
    ST_LOG_ERROR("st_inplace_add requires contiguous tensors.");
    return false;
  }
  ST_ASSERT_NOT_PENDING(a);
  ST_ASSERT_NOT_PENDING(b);

  ST_INPLACE_BINARY_BF16_PROMOTE(a, b, st_inplace_add);

#if defined(USE_ACCELERATE)
  cblas_saxpy((BLASINT)a->numel, 1.0f, b->values, 1, a->values, 1);
#else
#pragma omp parallel for schedule(static) if (a->numel > 10000)
  for (size_t i = 0; i < a->numel; ++i) {
    a->values[i] += b->values[i];
  }
#endif
  return true;
}

bool st_inplace_sub(FloatTensor *a, const FloatTensor *b) {
  if (a == NULL || a->values == NULL || b == NULL || b->values == NULL) {
    return false;
  }
  if (a->numel != b->numel) {
    ST_LOG_ERROR("st_inplace_sub size mismatch.");
    return false;
  }
  if (!st_is_contiguous(a) || !st_is_contiguous(b)) {
    ST_LOG_ERROR("st_inplace_sub requires contiguous tensors.");
    return false;
  }
  ST_ASSERT_NOT_PENDING(a);
  ST_ASSERT_NOT_PENDING(b);

  ST_INPLACE_BINARY_BF16_PROMOTE(a, b, st_inplace_sub);

#if defined(USE_ACCELERATE)
  cblas_saxpy((BLASINT)a->numel, -1.0f, b->values, 1, a->values, 1);
#else
#pragma omp parallel for schedule(static) if (a->numel > 10000)
  for (size_t i = 0; i < a->numel; ++i) {
    a->values[i] -= b->values[i];
  }
#endif
  return true;
}

bool st_inplace_scale(FloatTensor *t, float scalar) {
  if (t == NULL || t->values == NULL) {
    return false;
  }
  ST_ASSERT_NOT_PENDING(t);
  if (!st_is_contiguous(t)) {
    ST_LOG_ERROR("st_inplace_scale requires contiguous tensor.");
    return false;
  }

  /* ---- bf16 promotion ---- */
  if (t->dtype == ST_DTYPE_BF16) {
    FloatTensor *t_f32 = st_to_f32(t);
    if (!t_f32) return false;
    bool ok = st_inplace_scale(t_f32, scalar);
    if (ok) {
      st_f32_to_bf16_bulk(t_f32->values, (uint16_t *)t->values, t->numel);
    }
    st_destroy(t_f32);
    return ok;
  }

#if defined(USE_ACCELERATE)
  cblas_sscal((BLASINT)t->numel, scalar, t->values, 1);
#else
#pragma omp parallel for schedule(static) if (t->numel > 10000)
  for (size_t i = 0; i < t->numel; ++i) {
    t->values[i] *= scalar;
  }
#endif
  return true;
}

bool st_inplace_elementwise_multiply(FloatTensor *a, const FloatTensor *b) {
  if (a == NULL || a->values == NULL || b == NULL || b->values == NULL) {
    return false;
  }
  ST_ASSERT_NOT_PENDING(a);
  ST_ASSERT_NOT_PENDING(b);
  if (a->numel != b->numel) {
    ST_LOG_ERROR("st_inplace_elementwise_multiply size mismatch.");
    return false;
  }
  if (!st_is_contiguous(a) || !st_is_contiguous(b)) {
    ST_LOG_ERROR(
      "st_inplace_elementwise_multiply requires contiguous tensors.");
    return false;
  }

  ST_INPLACE_BINARY_BF16_PROMOTE(a, b, st_inplace_elementwise_multiply);

#if defined(USE_ACCELERATE)
  vDSP_vmul(a->values, 1, b->values, 1, a->values, 1,
            (vDSP_Length)a->numel);
#else
#pragma omp parallel for schedule(static) if (a->numel > 10000)
  for (size_t i = 0; i < a->numel; ++i) {
    a->values[i] *= b->values[i];
  }
#endif
  return true;
}

bool st_fill(FloatTensor *t, float value) {
  if (t == NULL || t->values == NULL) {
    return false;
  }
  ST_ASSERT_NOT_PENDING(t);
  if (!st_is_contiguous(t)) {
    ST_LOG_ERROR("st_fill requires contiguous tensor.");
    return false;
  }

  /* ---- bf16: fill with packed bf16 value ---- */
  if (t->dtype == ST_DTYPE_BF16) {
    uint16_t bf16_val = st_f32_to_bf16(value);
    uint16_t *dst = (uint16_t *)t->values;
    if (bf16_val == 0) {
      memset(dst, 0, t->numel * sizeof(uint16_t));
    } else {
      for (size_t i = 0; i < t->numel; ++i) {
        dst[i] = bf16_val;
      }
    }
    return true;
  }

  if (value == 0.0f) {
    memset(t->values, 0, t->numel * sizeof(float));
  } else {
#if defined(USE_ACCELERATE)
    vDSP_vfill(&value, t->values, 1, (vDSP_Length)t->numel);
#else
    for (size_t i = 0; i < t->numel; ++i) {
      t->values[i] = value;
    }
#endif
  }
  return true;
}

/* ---- Activation functions on tensors ---- */

bool st_apply_relu(FloatTensor *t) {
  if (t == NULL || t->values == NULL) {
    return false;
  }
  ST_ASSERT_NOT_PENDING(t);
  if (!st_is_contiguous(t)) {
    ST_LOG_ERROR("st_apply_relu requires contiguous tensor.");
    return false;
  }

  ST_INPLACE_UNARY_BF16_PROMOTE(t, st_apply_relu);

#if defined(USE_ACCELERATE)
  /* vDSP_vthr: t[i] = max(t[i], threshold) */
  float zero = 0.0f;
  vDSP_vthr(t->values, 1, &zero, t->values, 1, (vDSP_Length)t->numel);
#else
#pragma omp parallel for schedule(static) if (t->numel > 10000)
  for (size_t i = 0; i < t->numel; ++i) {
    if (t->values[i] < 0.0f) {
      t->values[i] = 0.0f;
    }
  }
#endif
  return true;
}

bool st_apply_relu_backward(const FloatTensor *activation, FloatTensor *grad) {
  if (activation == NULL || activation->values == NULL || grad == NULL ||
      grad->values == NULL) {
    return false;
  }
  ST_ASSERT_NOT_PENDING(activation);
  ST_ASSERT_NOT_PENDING(grad);
  if (activation->numel != grad->numel) {
    ST_LOG_ERROR("st_apply_relu_backward size mismatch.");
    return false;
  }
  if (!st_is_contiguous(activation) || !st_is_contiguous(grad)) {
    ST_LOG_ERROR("st_apply_relu_backward requires contiguous tensors.");
    return false;
  }

  ST_INPLACE_BINARY_BF16_PROMOTE(activation, grad, st_apply_relu_backward);

#pragma omp parallel for schedule(static) if (grad->numel > 10000)
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
  ST_ASSERT_NOT_PENDING(t);
  if (!st_is_contiguous(t)) {
    ST_LOG_ERROR("st_sum_axes requires contiguous tensor.");
    return NULL;
  }

  /* ---- bf16 promotion: reduce in f32 ---- */
  if (t->dtype == ST_DTYPE_BF16) {
    FloatTensor *t_f32 = st_to_f32(t);
    if (!t_f32) return NULL;
    FloatTensor *result = st_sum_axes(t_f32, axes, num_axes);
    st_destroy(t_f32);
    return result;
  }

  if (num_axes > t->ndim) {
    ST_LOG_ERROR("st_sum_axes more axes than dimensions.");
    return NULL;
  }

  /* Validate axes and build a reduction mask. */
  bool reduce[ST_MAX_DIMS] = {false};
  for (size_t i = 0; i < num_axes; ++i) {
    if (axes[i] >= t->ndim) {
      ST_LOG_ERROR("st_sum_axes axis out of range.");
      return NULL;
    }
    if (reduce[axes[i]]) {
      ST_LOG_ERROR("st_sum_axes duplicate axis.");
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

  /* ---- Fast path for common patterns ---- */

  /* Pattern: reduce axes {0, 2, 3} of 4D NCHW → [C] (bias gradient). */
  if (t->ndim == 4 && num_axes == 3 && reduce[0] && reduce[2] && reduce[3] &&
      !reduce[1]) {
    const size_t n = t->shape[0];
    const size_t c = t->shape[1];
    const size_t spatial = t->shape[2] * t->shape[3];

    for (size_t ci = 0; ci < c; ++ci) {
      float sum = 0.0f;
      for (size_t ni = 0; ni < n; ++ni) {
        const float *plane = t->values + (ni * c + ci) * spatial;
#if defined(USE_ACCELERATE)
        float plane_sum = 0.0f;
        vDSP_sve(plane, 1, &plane_sum, (vDSP_Length)spatial);
        sum += plane_sum;
#else
        for (size_t i = 0; i < spatial; ++i) {
          sum += plane[i];
        }
#endif
      }
      result->values[ci] = sum;
    }
    return result;
  }

  /* Pattern: reduce single leading axis of 2D [M, N] → [N]. */
  if (t->ndim == 2 && num_axes == 1 && reduce[0] && !reduce[1]) {
    const size_t rows = t->shape[0];
    const size_t cols = t->shape[1];
    for (size_t r = 0; r < rows; ++r) {
      const float *row = t->values + r * cols;
#if defined(USE_ACCELERATE)
      cblas_saxpy((BLASINT)cols, 1.0f, row, 1, result->values, 1);
#else
      for (size_t j = 0; j < cols; ++j) {
        result->values[j] += row[j];
      }
#endif
    }
    return result;
  }

  /* ---- Generic N-dimensional fallback (pre-compute strides) ---- */

  /* Precompute input strides to avoid repeated division/modulo. */
  size_t in_strides[ST_MAX_DIMS];
  in_strides[t->ndim - 1] = 1;
  for (size_t d = t->ndim - 1; d > 0; --d) {
    in_strides[d - 1] = in_strides[d] * t->shape[d];
  }

  /* Precompute output strides for non-reduced dims. */
  size_t out_strides[ST_MAX_DIMS] = {0};
  if (out_ndim > 0) {
    out_strides[out_ndim - 1] = 1;
    for (size_t d = out_ndim - 1; d > 0; --d) {
      out_strides[d - 1] = out_strides[d] * out_shape[d];
    }
  }

  /* Map: for each input dim that is NOT reduced, which output dim? */
  size_t dim_to_out[ST_MAX_DIMS] = {0};
  {
    size_t od = 0;
    for (size_t d = 0; d < t->ndim; ++d) {
      if (!reduce[d]) {
        dim_to_out[d] = od++;
      }
    }
  }

  size_t indices[ST_MAX_DIMS] = {0};

  for (size_t linear = 0; linear < t->numel; ++linear) {
    /* Decompose linear index via strides (no division). */
    size_t rem = linear;
    for (size_t d = 0; d < t->ndim; ++d) {
      indices[d] = rem / in_strides[d];
      rem %= in_strides[d];
    }

    /* Compute output linear index. */
    size_t out_linear = 0;
    for (size_t d = 0; d < t->ndim; ++d) {
      if (!reduce[d]) {
        out_linear += indices[d] * out_strides[dim_to_out[d]];
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
    ST_LOG_ERROR("st_pad_nchw expects valid 4D tensor.");
    return NULL;
  }
  ST_ASSERT_NOT_PENDING(input);
  if (!st_is_contiguous(input)) {
    ST_LOG_ERROR("st_pad_nchw requires contiguous tensor.");
    return NULL;
  }

  /* ---- bf16 promotion: pad in f32, convert result back ---- */
  if (input->dtype == ST_DTYPE_BF16) {
    FloatTensor *in_f32 = st_to_f32(input);
    if (!in_f32) return NULL;
    FloatTensor *result_f32 = st_pad_nchw(in_f32, pad_h, pad_w, value);
    st_destroy(in_f32);
    if (!result_f32) return NULL;
    FloatTensor *result = st_to_bf16(result_f32);
    st_destroy(result_f32);
    return result;
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

/* ============================================================================
 * Shape Transformation Operations (st_shape_ops.h)
 * ============================================================================ */

FloatTensor *st_flatten(const FloatTensor *tensor, size_t start_axis,
                        size_t end_axis) {
  if (tensor == NULL || start_axis > end_axis || end_axis > tensor->ndim) {
    return NULL;
  }

  /* Edge case: no flattening needed. */
  if (start_axis == end_axis) {
    return st_clone(tensor);
  }

  /* Compute merged dimension size. */
  size_t merge_size = 1;
  for (size_t i = start_axis; i < end_axis; ++i) {
    if (merge_size > SIZE_MAX / tensor->shape[i]) {
      return NULL;  /* overflow */
    }
    merge_size *= tensor->shape[i];
  }

  /* Build new shape: [0..start_axis) + merged + (end_axis..ndim) */
  size_t new_ndim = tensor->ndim - (end_axis - start_axis) + 1;
  if (new_ndim > ST_MAX_DIMS) {
    return NULL;
  }

  size_t new_shape[ST_MAX_DIMS];
  ptrdiff_t new_strides[ST_MAX_DIMS];

  /* Copy axes before start_axis. */
  for (size_t i = 0; i < start_axis; ++i) {
    new_shape[i] = tensor->shape[i];
    new_strides[i] = tensor->strides[i];
  }

  /* Insert merged axis. */
  new_shape[start_axis] = merge_size;
  /* Stride of merged axis is stride of innermost merged axis. */
  new_strides[start_axis] = tensor->strides[end_axis - 1];

  /* Copy axes after end_axis. */
  for (size_t i = end_axis; i < tensor->ndim; ++i) {
    new_shape[start_axis + 1 + (i - end_axis)] = tensor->shape[i];
    new_strides[start_axis + 1 + (i - end_axis)] = tensor->strides[i];
  }

  /* Create view. */
  return st_view((FloatTensor *)tensor, new_ndim, new_shape, new_strides, 0);
}

FloatTensor *st_flatten_all(const FloatTensor *tensor) {
  if (tensor == NULL) {
    return NULL;
  }
  return st_flatten(tensor, 0, tensor->ndim);
}

FloatTensor *st_permute(const FloatTensor *tensor, const size_t *axes) {
  if (tensor == NULL || axes == NULL) {
    return NULL;
  }
  return st_permute_view((FloatTensor *)tensor, axes);
}

FloatTensor *st_concat(const FloatTensor *const *tensors, size_t num_tensors,
                       size_t axis) {
  if (tensors == NULL || num_tensors == 0) {
    return NULL;
  }

  /* Validate all tensors. */
  for (size_t i = 0; i < num_tensors; ++i) {
    if (tensors[i] == NULL || tensors[i]->ndim == 0) {
      return NULL;
    }
  }

  /* All tensors must have same ndim and shape except on concat axis. */
  size_t ndim = tensors[0]->ndim;
  if (axis >= ndim) {
    return NULL;
  }

  for (size_t i = 1; i < num_tensors; ++i) {
    if (tensors[i]->ndim != ndim) {
      return NULL;
    }
    for (size_t d = 0; d < ndim; ++d) {
      if (d != axis && tensors[i]->shape[d] != tensors[0]->shape[d]) {
        return NULL;
      }
    }
  }

  /* Compute output shape. */
  size_t out_shape[ST_MAX_DIMS];
  memcpy(out_shape, tensors[0]->shape, ndim * sizeof(size_t));

  size_t concat_size = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    if (concat_size > SIZE_MAX - tensors[i]->shape[axis]) {
      return NULL;  /* overflow */
    }
    concat_size += tensors[i]->shape[axis];
  }
  out_shape[axis] = concat_size;

  /* Create output tensor (contiguous, same dtype as first input). */
  FloatTensor *result = NULL;
  if (tensors[0]->dtype == ST_DTYPE_BF16) {
    result = st_create_bf16(ndim, out_shape);
  } else {
    result = st_create(ndim, out_shape);
  }

  if (result == NULL) {
    return NULL;
  }

  /* Copy data from each tensor into result. */
  size_t offset_on_concat_axis = 0;

  for (size_t t = 0; t < num_tensors; ++t) {
    const FloatTensor *src = tensors[t];
    size_t src_elements = src->numel;

    /* For each element in src, compute its offset in result. */
    for (size_t elem = 0; elem < src_elements; ++elem) {
      /* Convert flat index to multi-index in src. */
      size_t multi_idx[ST_MAX_DIMS];
      size_t temp = elem;
      for (int d = (int)ndim - 1; d >= 0; --d) {
        multi_idx[d] = temp % src->shape[d];
        temp /= src->shape[d];
      }

      /* Adjust concat axis for result position. */
      multi_idx[axis] += offset_on_concat_axis;

      /* Compute flat index in result. */
      size_t result_idx = 0;
      size_t mult = 1;
      for (int d = (int)ndim - 1; d >= 0; --d) {
        result_idx += multi_idx[d] * mult;
        mult *= result->shape[d];
      }

      /* Copy element (respects dtype). */
      if (src->dtype == ST_DTYPE_BF16) {
        uint16_t bf16_val = ((uint16_t *)src->values)[elem];
        ((uint16_t *)result->values)[result_idx] = bf16_val;
      } else {
        result->values[result_idx] = src->values[elem];
      }
    }

    offset_on_concat_axis += src->shape[axis];
  }

  return result;
}
