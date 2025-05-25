/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "m_convert.h"

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
//
