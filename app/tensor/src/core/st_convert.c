/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "st_convert.h"

#include <log.h>
#include <string.h>

FloatTensor *st_from_sm(const FloatMatrix *src) {
  if (src == NULL || src->values == NULL || src->rows == 0 || src->cols == 0) {
    log_error("Error: st_from_sm received invalid matrix.");
    return NULL;
  }

  size_t shape[2] = {src->rows, src->cols};
  FloatTensor *tensor = st_create(2, shape);
  if (!tensor) {
    return NULL;
  }

  float *dst = st_tensor_mutable_data(tensor);
  if (!dst) {
    st_destroy(tensor);
    return NULL;
  }

  memcpy(dst, src->values, src->rows * src->cols * sizeof(float));
  return tensor;
}

FloatMatrix *sm_from_st(const FloatTensor *src) {
  if (src == NULL || st_tensor_ndim(src) != 2 || st_tensor_numel(src) == 0) {
    log_error("Error: sm_from_st requires a valid 2D tensor.");
    return NULL;
  }

  const size_t *shape = st_tensor_shape(src);
  if (!shape) {
    log_error("Error: sm_from_st missing tensor shape.");
    return NULL;
  }

  FloatMatrix *sm = sm_create(shape[0], shape[1]);
  if (!sm) {
    return NULL;
  }

  if (st_tensor_dtype(src) != ST_DTYPE_F32) {
    log_error("Error: sm_from_st currently supports only f32 tensors.");
    sm_destroy(sm);
    return NULL;
  }

  const float *src_data = st_tensor_data(src);
  const size_t rows = shape[0];
  const size_t cols = shape[1];

  if (src_data && st_is_contiguous(src)) {
    memcpy(sm->values, src_data, st_tensor_numel(src) * sizeof(float));
    return sm;
  }

  size_t idx[2] = {0, 0};
  for (size_t i = 0; i < rows; ++i) {
    idx[0] = i;
    for (size_t j = 0; j < cols; ++j) {
      idx[1] = j;
      sm->values[i * cols + j] = st_get(src, idx);
    }
  }

  return sm;
}
