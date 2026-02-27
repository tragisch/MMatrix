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
  if (!dms || !dms->values || !dms->row_indices || !dms->col_indices) {
    log_error("Error: dms_to_dm received invalid sparse matrix.");
    return NULL;
  }

  DoubleMatrix *dm = dm_create(dms->rows, dms->cols);
  if (!dm) {
    return NULL;
  }

  for (size_t k = 0; k < dms->nnz; ++k) {
    size_t i = dms->row_indices[k];
    size_t j = dms->col_indices[k];
    dm->values[i * dms->cols + j] = dms->values[k];
  }
  return dm;
}

DoubleSparseMatrix *dm_to_dms(const DoubleMatrix *dm) {
  if (!dm || !dm->values) {
    log_error("Error: dm_to_dms received invalid matrix.");
    return NULL;
  }

  // First pass: count non-zero elements
  size_t nnz = 0;
  size_t total = dm->rows * dm->cols;
  for (size_t i = 0; i < total; ++i) {
    if (dm->values[i] != 0.0) {
      ++nnz;
    }
  }

  DoubleSparseMatrix *dms = dms_create(dm->rows, dm->cols, nnz > 0 ? nnz : 1);
  if (!dms) {
    return NULL;
  }

  // Second pass: fill arrays directly (row-major order is already sorted)
  size_t k = 0;
  for (size_t i = 0; i < dm->rows; ++i) {
    for (size_t j = 0; j < dm->cols; ++j) {
      double val = dm->values[i * dm->cols + j];
      if (val != 0.0) {
        dms->row_indices[k] = i;
        dms->col_indices[k] = j;
        dms->values[k] = val;
        ++k;
      }
    }
  }
  dms->nnz = k;
  return dms;
}

DoubleMatrix *sm_to_dm(const FloatMatrix *sm) {
  DoubleMatrix *dm = dm_create(sm->rows, sm->cols);
  size_t total = sm->rows * sm->cols;
#pragma omp parallel for simd schedule(static) if (total > 10000)
  for (size_t i = 0; i < total; ++i) {
    dm->values[i] = (double)sm->values[i];
  }
  return dm;
}

FloatMatrix *dm_to_sm(const DoubleMatrix *dm) {
  FloatMatrix *sm = sm_create(dm->rows, dm->cols);
  size_t total = dm->rows * dm->cols;
#pragma omp parallel for simd schedule(static) if (total > 10000)
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
  ptrdiff_t stride0 = src->strides[0];
  ptrdiff_t stride1 = src->strides[1];

  // Validate strides once before the loop instead of per-element
  ptrdiff_t max_off = (ptrdiff_t)(rows - 1) * stride0 +
                      (ptrdiff_t)(cols - 1) * stride1;
  ptrdiff_t min_off = 0;
  // Check all four corners for negative strides
  ptrdiff_t corners[4] = {
      0,
      (ptrdiff_t)(rows - 1) * stride0,
      (ptrdiff_t)(cols - 1) * stride1,
      max_off,
  };
  for (int c = 0; c < 4; ++c) {
    if (corners[c] < min_off) min_off = corners[c];
    if (corners[c] > max_off) max_off = corners[c];
  }
  if (min_off < 0 || (size_t)max_off >= src->capacity) {
    sm_destroy(sm);
    log_error("Error: sm_from_st encountered invalid tensor strides.");
    return NULL;
  }

  for (size_t i = 0; i < rows; ++i) {
    ptrdiff_t row_off = (ptrdiff_t)i * stride0;
    for (size_t j = 0; j < cols; ++j) {
      sm->values[i * cols + j] = src->values[row_off + (ptrdiff_t)j * stride1];
    }
  }

  return sm;
}
//
