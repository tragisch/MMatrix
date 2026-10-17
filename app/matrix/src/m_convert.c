/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "m_convert.h"

#include <log.h>

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
