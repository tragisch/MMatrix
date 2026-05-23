/**
 * @file vv.h
 * @brief Public API for non-owning strided float vector views.
 */

#ifndef VV_H
#define VV_H

#include <stdbool.h>
#include <stddef.h>

typedef struct FloatVector FloatVector;

typedef struct FloatVectorView {
  size_t len;
  ptrdiff_t stride;
  float *values;
} FloatVectorView;

FloatVectorView vv_make(float *data, size_t len, ptrdiff_t stride);
bool vv_is_valid(const FloatVectorView *view);
float vv_get(const FloatVectorView *view, size_t index);
bool vv_set(FloatVectorView *view, size_t index, float value);
float vv_dot(const FloatVectorView *lhs, const FloatVectorView *rhs);
float vv_norm_l2(const FloatVectorView *view);
FloatVector *vv_to_sv(const FloatVectorView *view);
bool vv_copy_from_sv(FloatVectorView *dst, const FloatVector *src);

#endif  // VV_H
