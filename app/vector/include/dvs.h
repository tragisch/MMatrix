/**
 * @file dvs.h
 * @brief Public API for sparse double-precision vectors.
 */

#ifndef DVS_H
#define DVS_H

#include <stdbool.h>
#include <stddef.h>

typedef struct DoubleVector DoubleVector;

typedef struct DoubleSparseVector {
  size_t dim;
  size_t nnz;
  size_t capacity;
  size_t *indices;
  double *values;
} DoubleSparseVector;

DoubleSparseVector *dvs_create(size_t dim, size_t capacity);
DoubleSparseVector *dvs_clone(const DoubleSparseVector *vec);
bool dvs_set(DoubleSparseVector *vec, size_t index, double value);
double dvs_get(const DoubleSparseVector *vec, size_t index);
bool dvs_scale(DoubleSparseVector *vec, double scalar);
bool dvs_sort_indices(DoubleSparseVector *vec);
bool dvs_compact(DoubleSparseVector *vec);
DoubleSparseVector *dvs_add(const DoubleSparseVector *lhs,
                            const DoubleSparseVector *rhs);
double dvs_dot_dense(const DoubleSparseVector *lhs, const DoubleVector *rhs);
void dvs_destroy(DoubleSparseVector *vec);

#endif  // DVS_H
