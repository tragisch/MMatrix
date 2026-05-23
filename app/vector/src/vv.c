/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "vv.h"

#include "sv.h"

#include <limits.h>
#include <math.h>
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

static bool vv_has_compatible_shape(const FloatVectorView *lhs,
                                    const FloatVectorView *rhs) {
  return vv_is_valid(lhs) && vv_is_valid(rhs) && lhs->len == rhs->len &&
         lhs->stride > 0 && rhs->stride > 0;
}

FloatVectorView vv_make(float *data, size_t len, ptrdiff_t stride) {
  FloatVectorView view;
  view.len = len;
  view.stride = stride;
  view.values = data;
  return view;
}

bool vv_is_valid(const FloatVectorView *view) {
  return view != NULL && view->values != NULL && view->stride > 0;
}

float vv_get(const FloatVectorView *view, size_t index) {
  if (!vv_is_valid(view) || index >= view->len) {
    return 0.0f;
  }
  return view->values[index * (size_t)view->stride];
}

bool vv_set(FloatVectorView *view, size_t index, float value) {
  if (!vv_is_valid(view) || index >= view->len) {
    return false;
  }
  view->values[index * (size_t)view->stride] = value;
  return true;
}

float vv_dot(const FloatVectorView *lhs, const FloatVectorView *rhs) {
  if (!vv_has_compatible_shape(lhs, rhs)) {
    return 0.0f;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  if (lhs->stride <= INT_MAX && rhs->stride <= INT_MAX && lhs->len <= INT_MAX) {
    return cblas_sdot((BLASINT)lhs->len, lhs->values, (BLASINT)lhs->stride,
                      rhs->values, (BLASINT)rhs->stride);
  }
#endif
  float sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < lhs->len; ++i) {
    sum += lhs->values[i * (size_t)lhs->stride] *
           rhs->values[i * (size_t)rhs->stride];
  }
  return sum;
}

float vv_norm_l2(const FloatVectorView *view) {
  if (!vv_is_valid(view)) {
    return 0.0f;
  }
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  if (view->stride <= INT_MAX && view->len <= INT_MAX) {
    return cblas_snrm2((BLASINT)view->len, view->values, (BLASINT)view->stride);
  }
#endif
  float sum = 0.0f;
#pragma omp parallel for simd reduction(+ : sum)
  for (size_t i = 0; i < view->len; ++i) {
    const float value = view->values[i * (size_t)view->stride];
    sum += value * value;
  }
  return sqrtf(sum);
}

FloatVector *vv_to_sv(const FloatVectorView *view) {
  if (!vv_is_valid(view)) {
    return NULL;
  }
  FloatVector *vec = sv_create(view->len);
  if (vec == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < view->len; ++i) {
    vec->values[i] = view->values[i * (size_t)view->stride];
  }
  return vec;
}

bool vv_copy_from_sv(FloatVectorView *dst, const FloatVector *src) {
  if (!vv_is_valid(dst) || src == NULL || src->values == NULL ||
      dst->len != src->len) {
    return false;
  }
  for (size_t i = 0; i < dst->len; ++i) {
    dst->values[i * (size_t)dst->stride] = src->values[i];
  }
  return true;
}
