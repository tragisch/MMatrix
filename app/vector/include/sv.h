/**
 * @file sv.h
 * @brief Public API for dense single-precision vectors.
 */

#ifndef SV_H
#define SV_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum SvBackend {
  SV_BACKEND_DEFAULT = 0,
  SV_BACKEND_ACCELERATE = 1,
  SV_BACKEND_OPENBLAS = 2,
  SV_BACKEND_OPENMP = 3,
} SvBackend;

typedef struct FloatVector {
  size_t len;
  size_t capacity;
  float *values;
} FloatVector;

bool sv_set_backend(SvBackend backend);
SvBackend sv_get_backend(void);
const char *sv_active_library(void);

FloatVector *sv_create(size_t len);
FloatVector *sv_create_with_values(size_t len, const float *values);
FloatVector *sv_clone(const FloatVector *vec);
float *sv_to_array(const FloatVector *vec);

float sv_get(const FloatVector *vec, size_t index);
bool sv_set(FloatVector *vec, size_t index, float value);
bool sv_fill(FloatVector *vec, float value);

FloatVector *sv_add(const FloatVector *lhs, const FloatVector *rhs);
FloatVector *sv_sub(const FloatVector *lhs, const FloatVector *rhs);
FloatVector *sv_scale(const FloatVector *vec, float scalar);
bool sv_axpy(FloatVector *dst, float alpha, const FloatVector *src);

float sv_dot(const FloatVector *lhs, const FloatVector *rhs);
float sv_norm_l1(const FloatVector *vec);
float sv_norm_l2(const FloatVector *vec);
float sv_sum(const FloatVector *vec);
float sv_mean(const FloatVector *vec);
size_t sv_argmax(const FloatVector *vec);
bool sv_normalize(FloatVector *vec);
bool sv_softmax(FloatVector *vec);

void sv_destroy(FloatVector *vec);

#endif  // SV_H
