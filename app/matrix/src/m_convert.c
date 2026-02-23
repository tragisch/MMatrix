/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "m_convert.h"

#include <log.h>
#include <string.h>

DoubleMatrix *dms_to_dm(const DoubleSparseMatrix *dms) {
  DoubleMatrix *dm = dm_create(dms->rows, dms->cols);
  for (size_t k = 0; k < dms->nnz; ++k) {
    size_t i = dms->row_indices[k];
    size_t j = dms->col_indices[k];
    dm->values[i * dms->cols + j] = dms->values[k];
  }
  return dm;
}

DoubleSparseMatrix *dm_to_dms(const DoubleMatrix *dm) {
  size_t capacity = dm->rows * dm->cols;
  DoubleSparseMatrix *dms = dms_create(dm->rows, dm->cols, capacity);

  for (size_t i = 0; i < dm->rows; ++i) {
    for (size_t j = 0; j < dm->cols; ++j) {
      double val = dm->values[i * dm->cols + j];
      if (val != 0.0) {
        dms_set(dms, i, j, val);
      }
    }
  }
  return dms;
}

DoubleMatrix *sm_to_dm(const FloatMatrix *sm) {
  DoubleMatrix *dm = dm_create(sm->rows, sm->cols);
  size_t total = sm->rows * sm->cols;
#pragma omp simd
  for (size_t i = 0; i < total; ++i) {
    dm->values[i] = (double)sm->values[i];
  }
  return dm;
}

FloatMatrix *dm_to_sm(const DoubleMatrix *dm) {
  FloatMatrix *sm = sm_create(dm->rows, dm->cols);
  size_t total = dm->rows * dm->cols;
#pragma omp simd
  for (size_t i = 0; i < total; ++i) {
    sm->values[i] = (float)dm->values[i];
  }
  return sm;
}

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

  memcpy(tensor->values, src->values, src->rows * src->cols * sizeof(float));
  tensor->layout = ST_LAYOUT_CONTIGUOUS;
  return tensor;
}

FloatMatrix *sm_from_st(const FloatTensor *src) {
  if (src == NULL || src->values == NULL || src->ndim != 2 || src->numel == 0) {
    log_error("Error: sm_from_st requires a valid 2D tensor.");
    return NULL;
  }

  FloatMatrix *sm = sm_create(src->shape[0], src->shape[1]);
  if (!sm) {
    return NULL;
  }

  if (st_is_contiguous(src)) {
    memcpy(sm->values, src->values, src->numel * sizeof(float));
    return sm;
  }

  size_t rows = src->shape[0];
  size_t cols = src->shape[1];
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      ptrdiff_t off = (ptrdiff_t)i * src->strides[0] +
                      (ptrdiff_t)j * src->strides[1];
      if (off < 0 || (size_t)off >= src->capacity) {
        sm_destroy(sm);
        log_error("Error: sm_from_st encountered invalid tensor strides.");
        return NULL;
      }
      sm->values[i * cols + j] = src->values[off];
    }
  }

  return sm;
}
//
