/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "m_convert.h"

DoubleMatrix *dms_to_dm(const DoubleSparseMatrix *dms_create_empty) {
  DoubleMatrix *dm_create_empty =
      dm_create(dms_create_empty->rows, dms_create_empty->cols);
  for (int i = 0; i < dms_create_empty->rows; i++) {
    for (int j = 0; j < dms_create_empty->cols; j++) {
      dm_set(dm_create_empty, i, j, dms_get(dms_create_empty, i, j));
    }
  }
  return dm_create_empty;
}

DoubleSparseMatrix *dm_to_dms(const DoubleMatrix *dm_create_empty) {
  DoubleSparseMatrix *dms_create_empty =
      dms_create(dm_create_empty->rows, dm_create_empty->cols,
                 dm_create_empty->rows * dm_create_empty->cols);
  for (int i = 0; i < dm_create_empty->rows; i++) {
    for (int j = 0; j < dm_create_empty->cols; j++) {
      if (dm_get(dm_create_empty, i, j) != 0) {
        dms_set(dms_create_empty, i, j, dm_get(dm_create_empty, i, j));
      }
    }
  }
  return dms_create_empty;
}

DoubleMatrix *sm_to_dm(const FloatMatrix *sm) {
  DoubleMatrix *dm = dm_create(sm->rows, sm->cols);
  for (size_t i = 0; i < sm->rows; i++) {
    for (size_t j = 0; j < sm->cols; j++) {
      dm_set(dm, i, j, (double)sm_get(sm, i, j));
    }
  }
  return dm;
}

FloatMatrix *dm_to_sm(const DoubleMatrix *dm) {
  FloatMatrix *sm = sm_create(dm->rows, dm->cols);
  for (size_t i = 0; i < dm->rows; i++) {
    for (size_t j = 0; j < dm->cols; j++) {
      sm_set(sm, i, j, (float)dm_get(dm, i, j));
    }
  }
  return sm;
}
//