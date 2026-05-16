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
//
