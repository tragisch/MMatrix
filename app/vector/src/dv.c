/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "dv.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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

static const double DV_EPSILON = 1e-12;

static bool dv_has_compatible_shape(const DoubleVector *lhs,
                                    const DoubleVector *rhs) {
  return lhs != NULL && rhs != NULL && lhs->len == rhs->len &&
         lhs->values != NULL && rhs->values != NULL;
}

static DoubleVector *dv_create_uninitialized(size_t len) {
  DoubleVector *vec = (DoubleVector *)malloc(sizeof(DoubleVector));
  if (vec == NULL) {
    return NULL;
  }
  vec->len = len;
  vec->capacity = len;
  vec->values = len == 0 ? NULL : (double *)malloc(len * sizeof(double));
  if (len > 0 && vec->values == NULL) {
    free(vec);
    return NULL;
  }
  return vec;
}

DoubleVector *dv_create(size_t len) {
  DoubleVector *vec = dv_create_uninitialized(len);
  if (vec == NULL) {
    return NULL;
  }
  if (len > 0) {
    memset(vec->values, 0, len * sizeof(double));
  }
  return vec;
}

DoubleVector *dv_create_with_values(size_t len, const double *values) {
  if (len > 0 && values == NULL) {
    return NULL;
  }
  DoubleVector *vec = dv_create_uninitialized(len);
  if (vec == NULL) {
    return NULL;
  }
  if (len > 0) {
    memcpy(vec->values, values, len * sizeof(double));
  }
  return vec;
}

DoubleVector *dv_clone(const DoubleVector *vec) {
  if (vec == NULL) {
    return NULL;
  }
  return dv_create_with_values(vec->len, vec->values);
}

double *dv_to_array(const DoubleVector *vec) {
  if (vec == NULL || vec->len == 0 || vec->values == NULL) {
    return NULL;
  }
  double *copy = (double *)malloc(vec->len * sizeof(double));
  if (copy == NULL) {
    return NULL;
  }
  memcpy(copy, vec->values, vec->len * sizeof(double));
  return copy;
}

double dv_get(const DoubleVector *vec, size_t index) {
  return (vec == NULL || vec->values == NULL || index >= vec->len)
             ? 0.0
             : vec->values[index];
}

bool dv_set(DoubleVector *vec, size_t index, double value) {
  if (vec == NULL || vec->values == NULL || index >= vec->len) {
    return false;
  }
  vec->values[index] = value;
  return true;
}

bool dv_fill(DoubleVector *vec, double value) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return false;
  }
  for (size_t i = 0; i < vec->len; ++i) {
    vec->values[i] = value;
  }
  return true;
}

DoubleVector *dv_add(const DoubleVector *lhs, const DoubleVector *rhs) {
  if (!dv_has_compatible_shape(lhs, rhs)) {
    return NULL;
  }
  DoubleVector *out = dv_clone(lhs);
  if (out == NULL) {
    return NULL;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_daxpy((BLASINT)lhs->len, 1.0, rhs->values, 1, out->values, 1);
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < lhs->len; ++i) {
    out->values[i] += rhs->values[i];
  }
#endif
  return out;
}

DoubleVector *dv_sub(const DoubleVector *lhs, const DoubleVector *rhs) {
  if (!dv_has_compatible_shape(lhs, rhs)) {
    return NULL;
  }
  DoubleVector *out = dv_clone(lhs);
  if (out == NULL) {
    return NULL;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_daxpy((BLASINT)lhs->len, -1.0, rhs->values, 1, out->values, 1);
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < lhs->len; ++i) {
    out->values[i] -= rhs->values[i];
  }
#endif
  return out;
}

DoubleVector *dv_scale(const DoubleVector *vec, double scalar) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return NULL;
  }
  DoubleVector *out = dv_clone(vec);
  if (out == NULL) {
    return NULL;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_dscal((BLASINT)out->len, scalar, out->values, 1);
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < out->len; ++i) {
    out->values[i] *= scalar;
  }
#endif
  return out;
}

bool dv_axpy(DoubleVector *dst, double alpha, const DoubleVector *src) {
  if (!dv_has_compatible_shape(dst, src)) {
    return false;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_daxpy((BLASINT)dst->len, alpha, src->values, 1, dst->values, 1);
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < dst->len; ++i) {
    dst->values[i] += alpha * src->values[i];
  }
#endif
  return true;
}

double dv_dot(const DoubleVector *lhs, const DoubleVector *rhs) {
  if (!dv_has_compatible_shape(lhs, rhs)) {
    return 0.0;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  return cblas_ddot((BLASINT)lhs->len, lhs->values, 1, rhs->values, 1);
#else
  double sum = 0.0;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < lhs->len; ++i) {
    sum += lhs->values[i] * rhs->values[i];
  }
  return sum;
#endif
}

double dv_norm_l1(const DoubleVector *vec) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return 0.0;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  return cblas_dasum((BLASINT)vec->len, vec->values, 1);
#else
  double sum = 0.0;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < vec->len; ++i) {
    sum += fabs(vec->values[i]);
  }
  return sum;
#endif
}

double dv_norm_l2(const DoubleVector *vec) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return 0.0;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  return cblas_dnrm2((BLASINT)vec->len, vec->values, 1);
#else
  double sum = 0.0;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < vec->len; ++i) {
    sum += vec->values[i] * vec->values[i];
  }
  return sqrt(sum);
#endif
}

double dv_sum(const DoubleVector *vec) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return 0.0;
  }
  double sum = 0.0;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < vec->len; ++i) {
    sum += vec->values[i];
  }
  return sum;
}

double dv_mean(const DoubleVector *vec) {
  if (vec == NULL || vec->len == 0) {
    return 0.0;
  }
  return dv_sum(vec) / (double)vec->len;
}

size_t dv_argmax(const DoubleVector *vec) {
  if (vec == NULL || vec->len == 0 || vec->values == NULL) {
    return (size_t)-1;
  }
  size_t best = 0;
  for (size_t i = 1; i < vec->len; ++i) {
    if (vec->values[i] > vec->values[best]) {
      best = i;
    }
  }
  return best;
}

bool dv_normalize(DoubleVector *vec) {
  if (vec == NULL || vec->values == NULL) {
    return false;
  }
  double norm = dv_norm_l2(vec);
  if (norm <= DV_EPSILON) {
    return false;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_dscal((BLASINT)vec->len, 1.0 / norm, vec->values, 1);
#else
  double inv = 1.0 / norm;
#pragma omp parallel for simd
  for (size_t i = 0; i < vec->len; ++i) {
    vec->values[i] *= inv;
  }
#endif
  return true;
}

void dv_destroy(DoubleVector *vec) {
  if (vec == NULL) {
    return;
  }
  free(vec->values);
  free(vec);
}
