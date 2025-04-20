#include "dm_convert.h"

DoubleMatrix *dms_to_dm(const DoubleSparseMatrix *dms) {
  DoubleMatrix *dm_create_empty = dm_create(dms->rows, dms->cols);
  for (int i = 0; i < dms->rows; i++) {
    for (int j = 0; j < dms->cols; j++) {
      dm_set(dm_create_empty, i, j, dms_get(dms, i, j));
    }
  }
  return dm_create_empty;
}

DoubleSparseMatrix *dm_to_dms(const DoubleMatrix *dm_create_empty) {
  DoubleSparseMatrix *dms =
      dms_create(dm_create_empty->rows, dm_create_empty->cols,
                 dm_create_empty->rows * dm_create_empty->cols);
  for (int i = 0; i < dm_create_empty->rows; i++) {
    for (int j = 0; j < dm_create_empty->cols; j++) {
      if (dm_get(dm_create_empty, i, j) != 0) {
        dms_set(dms, i, j, dm_get(dm_create_empty, i, j));
      }
    }
  }
  return dms;
}