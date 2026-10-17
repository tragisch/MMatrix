/**
 * @file dv.h
 * @brief Public API for dense double-precision vectors.
 */

#ifndef DV_H
#define DV_H

#include <stdbool.h>
#include <stddef.h>

typedef struct DoubleVector {
  size_t len;
  size_t capacity;
  double *values;
} DoubleVector;

DoubleVector *dv_create(size_t len);
DoubleVector *dv_create_with_values(size_t len, const double *values);
DoubleVector *dv_clone(const DoubleVector *vec);
double *dv_to_array(const DoubleVector *vec);

double dv_get(const DoubleVector *vec, size_t index);
bool dv_set(DoubleVector *vec, size_t index, double value);
bool dv_fill(DoubleVector *vec, double value);

DoubleVector *dv_add(const DoubleVector *lhs, const DoubleVector *rhs);
DoubleVector *dv_sub(const DoubleVector *lhs, const DoubleVector *rhs);
DoubleVector *dv_scale(const DoubleVector *vec, double scalar);
bool dv_axpy(DoubleVector *dst, double alpha, const DoubleVector *src);

double dv_dot(const DoubleVector *lhs, const DoubleVector *rhs);
double dv_norm_l1(const DoubleVector *vec);
double dv_norm_l2(const DoubleVector *vec);
double dv_sum(const DoubleVector *vec);
double dv_mean(const DoubleVector *vec);
size_t dv_argmax(const DoubleVector *vec);
bool dv_normalize(DoubleVector *vec);

void dv_destroy(DoubleVector *vec);

#endif  // DV_H
