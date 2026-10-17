/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "vector_bridge.h"

#include <stddef.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpedantic"
#endif
#include <omp.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(USE_ACCELERATE)
#define BLASINT int
#include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
#define BLASINT int
#include <cblas.h>
#endif

FloatVectorView sm_row_view(FloatMatrix *mat, size_t row) {
  if (mat == NULL || mat->values == NULL || row >= mat->rows) {
    return vv_make(NULL, 0, 0);
  }
  return vv_make(&mat->values[row * mat->cols], mat->cols, 1);
}

FloatVectorView sm_col_view(FloatMatrix *mat, size_t col) {
  if (mat == NULL || mat->values == NULL || col >= mat->cols) {
    return vv_make(NULL, 0, 0);
  }
  return vv_make(&mat->values[col], mat->rows, (ptrdiff_t)mat->cols);
}

FloatVector *sm_row_to_sv(const FloatMatrix *mat, size_t row) {
  if (mat == NULL || mat->values == NULL || row >= mat->rows) {
    return NULL;
  }
  return sv_create_with_values(mat->cols, &mat->values[row * mat->cols]);
}

FloatVector *sm_col_to_sv(const FloatMatrix *mat, size_t col) {
  if (mat == NULL || mat->values == NULL || col >= mat->cols) {
    return NULL;
  }
  FloatVector *vec = sv_create(mat->rows);
  if (vec == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < mat->rows; ++i) {
    vec->values[i] = mat->values[i * mat->cols + col];
  }
  return vec;
}

FloatVector *sm_matvec(const FloatMatrix *mat, const FloatVector *vec) {
  if (mat == NULL || vec == NULL || mat->values == NULL || vec->values == NULL ||
      mat->cols != vec->len) {
    return NULL;
  }
  FloatVector *out = sv_create(mat->rows);
  if (out == NULL) {
    return NULL;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_sgemv(CblasRowMajor, CblasNoTrans, (BLASINT)mat->rows,
              (BLASINT)mat->cols, 1.0f, mat->values, (BLASINT)mat->cols,
              vec->values, 1, 0.0f, out->values, 1);
#else
#pragma omp parallel for
  for (size_t i = 0; i < mat->rows; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < mat->cols; ++j) {
      sum += mat->values[i * mat->cols + j] * vec->values[j];
    }
    out->values[i] = sum;
  }
#endif
  return out;
}

FloatMatrix *sv_outer_as_sm(const FloatVector *lhs, const FloatVector *rhs) {
  if (lhs == NULL || rhs == NULL || lhs->values == NULL || rhs->values == NULL) {
    return NULL;
  }
  FloatMatrix *out = sm_create(lhs->len, rhs->len);
  if (out == NULL) {
    return NULL;
  }
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < lhs->len; ++i) {
    for (size_t j = 0; j < rhs->len; ++j) {
      out->values[i * rhs->len + j] = lhs->values[i] * rhs->values[j];
    }
  }
  return out;
}

DoubleVector *dm_row_to_dv(const DoubleMatrix *mat, size_t row) {
  if (mat == NULL || mat->values == NULL || row >= mat->rows) {
    return NULL;
  }
  return dv_create_with_values(mat->cols, &mat->values[row * mat->cols]);
}

DoubleVector *dm_col_to_dv(const DoubleMatrix *mat, size_t col) {
  if (mat == NULL || mat->values == NULL || col >= mat->cols) {
    return NULL;
  }
  DoubleVector *vec = dv_create(mat->rows);
  if (vec == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < mat->rows; ++i) {
    vec->values[i] = mat->values[i * mat->cols + col];
  }
  return vec;
}

DoubleVector *dm_matvec(const DoubleMatrix *mat, const DoubleVector *vec) {
  if (mat == NULL || vec == NULL || mat->values == NULL || vec->values == NULL ||
      mat->cols != vec->len) {
    return NULL;
  }
  DoubleVector *out = dv_create(mat->rows);
  if (out == NULL) {
    return NULL;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_dgemv(CblasRowMajor, CblasNoTrans, (BLASINT)mat->rows,
              (BLASINT)mat->cols, 1.0, mat->values, (BLASINT)mat->cols,
              vec->values, 1, 0.0, out->values, 1);
#else
#pragma omp parallel for
  for (size_t i = 0; i < mat->rows; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < mat->cols; ++j) {
      sum += mat->values[i * mat->cols + j] * vec->values[j];
    }
    out->values[i] = sum;
  }
#endif
  return out;
}

DoubleMatrix *dv_outer_as_dm(const DoubleVector *lhs, const DoubleVector *rhs) {
  if (lhs == NULL || rhs == NULL || lhs->values == NULL || rhs->values == NULL) {
    return NULL;
  }
  DoubleMatrix *out = dm_create(lhs->len, rhs->len);
  if (out == NULL) {
    return NULL;
  }
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < lhs->len; ++i) {
    for (size_t j = 0; j < rhs->len; ++j) {
      out->values[i * rhs->len + j] = lhs->values[i] * rhs->values[j];
    }
  }
  return out;
}

DoubleSparseVector *dms_row_to_dvs(const DoubleSparseMatrix *mat, size_t row) {
  if (mat == NULL || row >= mat->rows) {
    return NULL;
  }
  DoubleSparseVector *vec = dvs_create(mat->cols, 0);
  if (vec == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < mat->nnz; ++i) {
    if (mat->row_indices[i] == row && !dvs_set(vec, mat->col_indices[i], mat->values[i])) {
      dvs_destroy(vec);
      return NULL;
    }
  }
  if (!dvs_compact(vec)) {
    dvs_destroy(vec);
    return NULL;
  }
  return vec;
}

DoubleSparseVector *dms_col_to_dvs(const DoubleSparseMatrix *mat, size_t col) {
  if (mat == NULL || col >= mat->cols) {
    return NULL;
  }
  DoubleSparseVector *vec = dvs_create(mat->rows, 0);
  if (vec == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < mat->nnz; ++i) {
    if (mat->col_indices[i] == col && !dvs_set(vec, mat->row_indices[i], mat->values[i])) {
      dvs_destroy(vec);
      return NULL;
    }
  }
  if (!dvs_compact(vec)) {
    dvs_destroy(vec);
    return NULL;
  }
  return vec;
}
