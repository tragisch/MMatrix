/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "sv.h"

#include <log.h>
#include <math.h>
#include <stddef.h>
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

static const float SV_EPSILON = 1e-8f;
static SvBackend sv_current_backend = SV_BACKEND_DEFAULT;

static bool sv_has_compatible_shape(const FloatVector *lhs,
                                    const FloatVector *rhs) {
  return lhs != NULL && rhs != NULL && lhs->len == rhs->len &&
         lhs->values != NULL && rhs->values != NULL;
}

static FloatVector *sv_create_uninitialized(size_t len) {
  FloatVector *vec = (FloatVector *)malloc(sizeof(FloatVector));
  if (vec == NULL) {
    log_error("Failed to allocate float vector metadata");
    return NULL;
  }
  vec->len = len;
  vec->capacity = len;
  vec->values = len == 0 ? NULL : (float *)malloc(len * sizeof(float));
  if (len > 0 && vec->values == NULL) {
    log_error("Failed to allocate float vector values");
    free(vec);
    return NULL;
  }
  return vec;
}

bool sv_set_backend(SvBackend backend) {
  switch (backend) {
    case SV_BACKEND_DEFAULT:
      sv_current_backend = backend;
      return true;
    case SV_BACKEND_ACCELERATE:
#if defined(USE_ACCELERATE)
      sv_current_backend = backend;
      return true;
#else
      return false;
#endif
    case SV_BACKEND_OPENBLAS:
#if defined(USE_OPENBLAS)
      sv_current_backend = backend;
      return true;
#else
      return false;
#endif
    case SV_BACKEND_OPENMP:
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
      return false;
#else
      sv_current_backend = backend;
      return true;
#endif
  }
  return false;
}

SvBackend sv_get_backend(void) { return sv_current_backend; }

const char *sv_active_library(void) {
  switch (sv_current_backend) {
    case SV_BACKEND_ACCELERATE:
#if defined(USE_ACCELERATE)
      return "Apple Accelerate";
#endif
      break;
    case SV_BACKEND_OPENBLAS:
#if defined(USE_OPENBLAS)
      return "OpenBLAS";
#endif
      break;
    case SV_BACKEND_OPENMP:
      return "OpenMP";
    case SV_BACKEND_DEFAULT:
    default:
      break;
  }
#if defined(USE_ACCELERATE)
  return "Apple Accelerate";
#elif defined(USE_OPENBLAS)
  return "OpenBLAS";
#else
  return "OpenMP";
#endif
}

FloatVector *sv_create(size_t len) {
  FloatVector *vec = sv_create_uninitialized(len);
  if (vec == NULL) {
    return NULL;
  }
  if (len > 0) {
    memset(vec->values, 0, len * sizeof(float));
  }
  return vec;
}

FloatVector *sv_create_with_values(size_t len, const float *values) {
  if (len > 0 && values == NULL) {
    return NULL;
  }
  FloatVector *vec = sv_create_uninitialized(len);
  if (vec == NULL) {
    return NULL;
  }
  if (len > 0) {
    memcpy(vec->values, values, len * sizeof(float));
  }
  return vec;
}

FloatVector *sv_clone(const FloatVector *vec) {
  if (vec == NULL) {
    return NULL;
  }
  return sv_create_with_values(vec->len, vec->values);
}

float *sv_to_array(const FloatVector *vec) {
  if (vec == NULL) {
    return NULL;
  }
  if (vec->len == 0) {
    return NULL;
  }
  float *copy = (float *)malloc(vec->len * sizeof(float));
  if (copy == NULL) {
    return NULL;
  }
  memcpy(copy, vec->values, vec->len * sizeof(float));
  return copy;
}

float sv_get(const FloatVector *vec, size_t index) {
  return (vec == NULL || vec->values == NULL || index >= vec->len)
             ? 0.0f
             : vec->values[index];
}

bool sv_set(FloatVector *vec, size_t index, float value) {
  if (vec == NULL || vec->values == NULL || index >= vec->len) {
    return false;
  }
  vec->values[index] = value;
  return true;
}

bool sv_fill(FloatVector *vec, float value) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return false;
  }
  for (size_t i = 0; i < vec->len; ++i) {
    vec->values[i] = value;
  }
  return true;
}

FloatVector *sv_add(const FloatVector *lhs, const FloatVector *rhs) {
  if (!sv_has_compatible_shape(lhs, rhs)) {
    return NULL;
  }
  FloatVector *out = sv_create(lhs->len);
  if (out == NULL) {
    return NULL;
  }
  memcpy(out->values, lhs->values, lhs->len * sizeof(float));
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_saxpy((BLASINT)lhs->len, 1.0f, rhs->values, 1, out->values, 1);
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < lhs->len; ++i) {
    out->values[i] += rhs->values[i];
  }
#endif
  return out;
}

FloatVector *sv_sub(const FloatVector *lhs, const FloatVector *rhs) {
  if (!sv_has_compatible_shape(lhs, rhs)) {
    return NULL;
  }
  FloatVector *out = sv_create(lhs->len);
  if (out == NULL) {
    return NULL;
  }
  memcpy(out->values, lhs->values, lhs->len * sizeof(float));
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_saxpy((BLASINT)lhs->len, -1.0f, rhs->values, 1, out->values, 1);
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < lhs->len; ++i) {
    out->values[i] -= rhs->values[i];
  }
#endif
  return out;
}

FloatVector *sv_scale(const FloatVector *vec, float scalar) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return NULL;
  }
  FloatVector *out = sv_clone(vec);
  if (out == NULL) {
    return NULL;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_sscal((BLASINT)out->len, scalar, out->values, 1);
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < out->len; ++i) {
    out->values[i] *= scalar;
  }
#endif
  return out;
}

bool sv_axpy(FloatVector *dst, float alpha, const FloatVector *src) {
  if (!sv_has_compatible_shape(dst, src)) {
    return false;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  cblas_saxpy((BLASINT)dst->len, alpha, src->values, 1, dst->values, 1);
#else
#pragma omp parallel for simd
  for (size_t i = 0; i < dst->len; ++i) {
    dst->values[i] += alpha * src->values[i];
  }
#endif
  return true;
}

float sv_dot(const FloatVector *lhs, const FloatVector *rhs) {
  if (!sv_has_compatible_shape(lhs, rhs)) {
    return 0.0f;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  return cblas_sdot((BLASINT)lhs->len, lhs->values, 1, rhs->values, 1);
#else
  float sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < lhs->len; ++i) {
    sum += lhs->values[i] * rhs->values[i];
  }
  return sum;
#endif
}

float sv_norm_l1(const FloatVector *vec) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return 0.0f;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  return cblas_sasum((BLASINT)vec->len, vec->values, 1);
#else
  float sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < vec->len; ++i) {
    sum += fabsf(vec->values[i]);
  }
  return sum;
#endif
}

float sv_norm_l2(const FloatVector *vec) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return 0.0f;
  }
#if defined(USE_ACCELERATE)
  return cblas_snrm2((BLASINT)vec->len, vec->values, 1);
#elif defined(USE_OPENBLAS)
  return cblas_snrm2((BLASINT)vec->len, vec->values, 1);
#else
  float sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < vec->len; ++i) {
    sum += vec->values[i] * vec->values[i];
  }
  return sqrtf(sum);
#endif
}

float sv_sum(const FloatVector *vec) {
  if (vec == NULL || (vec->len > 0 && vec->values == NULL)) {
    return 0.0f;
  }
#if defined(USE_ACCELERATE)
  float sum = 0.0f;
  vDSP_sve(vec->values, 1, &sum, (vDSP_Length)vec->len);
  return sum;
#else
  float sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < vec->len; ++i) {
    sum += vec->values[i];
  }
  return sum;
#endif
}

float sv_mean(const FloatVector *vec) {
  if (vec == NULL || vec->len == 0) {
    return 0.0f;
  }
  return sv_sum(vec) / (float)vec->len;
}

size_t sv_argmax(const FloatVector *vec) {
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

bool sv_normalize(FloatVector *vec) {
  if (vec == NULL || vec->values == NULL) {
    return false;
  }
  float norm = sv_norm_l2(vec);
  if (norm <= SV_EPSILON) {
    return false;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  float scale = 1.0f / norm;
  cblas_sscal((BLASINT)vec->len, scale, vec->values, 1);
#else
  float inv = 1.0f / norm;
#pragma omp parallel for simd
  for (size_t i = 0; i < vec->len; ++i) {
    vec->values[i] *= inv;
  }
#endif
  return true;
}

bool sv_softmax(FloatVector *vec) {
  if (vec == NULL || vec->len == 0 || vec->values == NULL) {
    return false;
  }
  size_t max_index = sv_argmax(vec);
  if (max_index == (size_t)-1) {
    return false;
  }
  float max_value = vec->values[max_index];
  float sum = 0.0f;
  for (size_t i = 0; i < vec->len; ++i) {
    vec->values[i] = expf(vec->values[i] - max_value);
    sum += vec->values[i];
  }
  if (sum <= SV_EPSILON) {
    return false;
  }
#pragma omp parallel for simd
  for (size_t i = 0; i < vec->len; ++i) {
    vec->values[i] /= sum;
  }
  return true;
}

void sv_destroy(FloatVector *vec) {
  if (vec == NULL) {
    return;
  }
  free(vec->values);
  free(vec);
}
