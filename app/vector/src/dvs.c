/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "dvs.h"

#include "dv.h"

#include <stdlib.h>
#include <string.h>

typedef struct DvsPair {
  size_t index;
  double value;
} DvsPair;

static int dvs_pair_compare(const void *lhs, const void *rhs) {
  const DvsPair *a = (const DvsPair *)lhs;
  const DvsPair *b = (const DvsPair *)rhs;
  if (a->index < b->index) {
    return -1;
  }
  if (a->index > b->index) {
    return 1;
  }
  return 0;
}

static bool dvs_ensure_capacity(DoubleSparseVector *vec, size_t min_capacity) {
  if (vec == NULL) {
    return false;
  }
  if (vec->capacity >= min_capacity) {
    return true;
  }
  size_t new_capacity = vec->capacity == 0 ? 1 : vec->capacity;
  while (new_capacity < min_capacity) {
    new_capacity *= 2;
  }
  size_t *new_indices =
      (size_t *)realloc(vec->indices, new_capacity * sizeof(size_t));
  if (new_indices == NULL) {
    return false;
  }
  double *new_values =
      (double *)realloc(vec->values, new_capacity * sizeof(double));
  if (new_values == NULL) {
    vec->indices = new_indices;
    return false;
  }
  vec->indices = new_indices;
  vec->values = new_values;
  vec->capacity = new_capacity;
  return true;
}

DoubleSparseVector *dvs_create(size_t dim, size_t capacity) {
  DoubleSparseVector *vec =
      (DoubleSparseVector *)calloc(1, sizeof(DoubleSparseVector));
  if (vec == NULL) {
    return NULL;
  }
  vec->dim = dim;
  vec->capacity = capacity;
  if (capacity > 0) {
    vec->indices = (size_t *)malloc(capacity * sizeof(size_t));
    vec->values = (double *)malloc(capacity * sizeof(double));
    if (vec->indices == NULL || vec->values == NULL) {
      free(vec->indices);
      free(vec->values);
      free(vec);
      return NULL;
    }
  }
  return vec;
}

DoubleSparseVector *dvs_clone(const DoubleSparseVector *vec) {
  if (vec == NULL) {
    return NULL;
  }
  DoubleSparseVector *copy = dvs_create(vec->dim, vec->nnz);
  if (copy == NULL) {
    return NULL;
  }
  copy->nnz = vec->nnz;
  if (vec->nnz > 0) {
    memcpy(copy->indices, vec->indices, vec->nnz * sizeof(size_t));
    memcpy(copy->values, vec->values, vec->nnz * sizeof(double));
  }
  return copy;
}

bool dvs_set(DoubleSparseVector *vec, size_t index, double value) {
  if (vec == NULL || index >= vec->dim) {
    return false;
  }
  for (size_t i = 0; i < vec->nnz; ++i) {
    if (vec->indices[i] == index) {
      if (value == 0.0) {
        if (i + 1 < vec->nnz) {
          memmove(&vec->indices[i], &vec->indices[i + 1],
                  (vec->nnz - i - 1) * sizeof(size_t));
          memmove(&vec->values[i], &vec->values[i + 1],
                  (vec->nnz - i - 1) * sizeof(double));
        }
        vec->nnz--;
        return true;
      }
      vec->values[i] = value;
      return true;
    }
  }
  if (value == 0.0) {
    return true;
  }
  if (!dvs_ensure_capacity(vec, vec->nnz + 1)) {
    return false;
  }
  vec->indices[vec->nnz] = index;
  vec->values[vec->nnz] = value;
  vec->nnz++;
  return true;
}

double dvs_get(const DoubleSparseVector *vec, size_t index) {
  if (vec == NULL || index >= vec->dim) {
    return 0.0;
  }
  for (size_t i = 0; i < vec->nnz; ++i) {
    if (vec->indices[i] == index) {
      return vec->values[i];
    }
  }
  return 0.0;
}

bool dvs_scale(DoubleSparseVector *vec, double scalar) {
  if (vec == NULL) {
    return false;
  }
  for (size_t i = 0; i < vec->nnz; ++i) {
    vec->values[i] *= scalar;
  }
  return true;
}

bool dvs_sort_indices(DoubleSparseVector *vec) {
  if (vec == NULL || vec->nnz < 2) {
    return vec != NULL;
  }
  DvsPair *pairs = (DvsPair *)malloc(vec->nnz * sizeof(DvsPair));
  if (pairs == NULL) {
    return false;
  }
  for (size_t i = 0; i < vec->nnz; ++i) {
    pairs[i].index = vec->indices[i];
    pairs[i].value = vec->values[i];
  }
  qsort(pairs, vec->nnz, sizeof(DvsPair), dvs_pair_compare);
  for (size_t i = 0; i < vec->nnz; ++i) {
    vec->indices[i] = pairs[i].index;
    vec->values[i] = pairs[i].value;
  }
  free(pairs);
  return true;
}

bool dvs_compact(DoubleSparseVector *vec) {
  if (vec == NULL) {
    return false;
  }
  if (!dvs_sort_indices(vec)) {
    return false;
  }
  size_t write = 0;
  for (size_t i = 0; i < vec->nnz; ++i) {
    if (write > 0 && vec->indices[write - 1] == vec->indices[i]) {
      vec->values[write - 1] += vec->values[i];
      continue;
    }
    vec->indices[write] = vec->indices[i];
    vec->values[write] = vec->values[i];
    write++;
  }
  size_t compacted = 0;
  for (size_t i = 0; i < write; ++i) {
    if (vec->values[i] != 0.0) {
      vec->indices[compacted] = vec->indices[i];
      vec->values[compacted] = vec->values[i];
      compacted++;
    }
  }
  vec->nnz = compacted;
  return true;
}

DoubleSparseVector *dvs_add(const DoubleSparseVector *lhs,
                            const DoubleSparseVector *rhs) {
  if (lhs == NULL || rhs == NULL || lhs->dim != rhs->dim) {
    return NULL;
  }
  DoubleSparseVector *out = dvs_clone(lhs);
  if (out == NULL) {
    return NULL;
  }
  if (!dvs_ensure_capacity(out, lhs->nnz + rhs->nnz)) {
    dvs_destroy(out);
    return NULL;
  }
  for (size_t i = 0; i < rhs->nnz; ++i) {
    if (!dvs_set(out, rhs->indices[i], dvs_get(out, rhs->indices[i]) + rhs->values[i])) {
      dvs_destroy(out);
      return NULL;
    }
  }
  if (!dvs_compact(out)) {
    dvs_destroy(out);
    return NULL;
  }
  return out;
}

double dvs_dot_dense(const DoubleSparseVector *lhs, const DoubleVector *rhs) {
  if (lhs == NULL || rhs == NULL || rhs->values == NULL || lhs->dim != rhs->len) {
    return 0.0;
  }
  double sum = 0.0;
  for (size_t i = 0; i < lhs->nnz; ++i) {
    sum += lhs->values[i] * rhs->values[lhs->indices[i]];
  }
  return sum;
}

void dvs_destroy(DoubleSparseVector *vec) {
  if (vec == NULL) {
    return;
  }
  free(vec->indices);
  free(vec->values);
  free(vec);
}
