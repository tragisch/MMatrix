/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "../include/dms.h"

#if defined(__has_include)
#if __has_include(<cs.h>) && __has_include(<omp.h>) && \
    __has_include(<pcg_variants.h>)
#define DMS_HAS_FULL_DEPS 1
#endif
#endif

#if defined(DMS_HAS_FULL_DEPS)

#include <cs.h>
#include <limits.h>
#include <log.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef INIT_CAPACITY
#define INIT_CAPACITY 100
#endif
#ifndef EPSILON
#define EPSILON 1e-9
#endif

// Shared RNG state used by non-seeded creators. Access is intentionally simple
// and remains process-global, so callers must provide external synchronization
// when reading or writing this state from multiple threads.
static uint64_t dms_global_seed = 0;
static bool dms_seed_initialized = false;

typedef struct dms_index_storage {
  bool is_sorted;
  bool builder_mode;
  size_t reserved;
  size_t data[];
} dms_index_storage;

typedef struct dms_sort_context {
  const size_t *rows;
  const size_t *cols;
} dms_sort_context;

/*******************************/
/*       Private Functions     */
/*******************************/

double dms_max_double(double a, double b) { return a > b ? a : b; }
double dms_min_double(double a, double b) { return a < b ? a : b; }
int dms_max_int(int a, int b) { return a > b ? a : b; }

static dms_index_storage *dms_index_storage_from_rows(
    const size_t *row_indices) {
  if (!row_indices) {
    return NULL;
  }
  return (dms_index_storage *)((char *)row_indices -
                               offsetof(dms_index_storage, data));
}

static bool dms_matrix_is_sorted(const DoubleSparseMatrix *mat) {
  dms_index_storage *storage =
      mat ? dms_index_storage_from_rows(mat->row_indices) : NULL;
  return !storage || storage->is_sorted;
}

static bool dms_matrix_builder_mode(const DoubleSparseMatrix *mat) {
  dms_index_storage *storage =
      mat ? dms_index_storage_from_rows(mat->row_indices) : NULL;
  return storage ? storage->builder_mode : false;
}

static void dms_matrix_set_state(DoubleSparseMatrix *mat, bool is_sorted,
                                 bool builder_mode) {
  dms_index_storage *storage =
      mat ? dms_index_storage_from_rows(mat->row_indices) : NULL;
  if (!storage) {
    return;
  }
  storage->is_sorted = is_sorted;
  storage->builder_mode = builder_mode;
}

static void dms_matrix_set_sorted(DoubleSparseMatrix *mat, bool is_sorted) {
  dms_index_storage *storage =
      mat ? dms_index_storage_from_rows(mat->row_indices) : NULL;
  if (!storage) {
    return;
  }
  storage->is_sorted = is_sorted;
}

static bool dms_allocate_storage(DoubleSparseMatrix *mat, size_t capacity,
                                 bool is_sorted, bool builder_mode) {
  dms_index_storage *index_storage = NULL;
  double *values = NULL;

  if (!mat || capacity == 0) {
    return false;
  }

  index_storage =
      malloc(sizeof(dms_index_storage) + (capacity * 2 * sizeof(size_t)));
  if (!index_storage) {
    log_error("Error allocating memory for sparse matrix indices");
    return false;
  }

  values = malloc(capacity * sizeof(double));
  if (!values) {
    log_error("Error allocating memory for sparse matrix values");
    free(index_storage);
    return false;
  }

  index_storage->is_sorted = is_sorted;
  index_storage->builder_mode = builder_mode;
  index_storage->reserved = 0;

  mat->row_indices = index_storage->data;
  mat->col_indices = index_storage->data + capacity;
  mat->values = values;

  return true;
}

static void dms_free_storage(DoubleSparseMatrix *mat) {
  if (!mat) {
    return;
  }

  if (mat->row_indices) {
    free(dms_index_storage_from_rows(mat->row_indices));
  }
  free(mat->values);

  mat->row_indices = NULL;
  mat->col_indices = NULL;
  mat->values = NULL;
}

static int dms_compare_coordinates(size_t row_a, size_t col_a, size_t row_b,
                                   size_t col_b) {
  if (row_a != row_b) {
    return (row_a < row_b) ? -1 : 1;
  }
  if (col_a != col_b) {
    return (col_a < col_b) ? -1 : 1;
  }
  return 0;
}

static bool dms_is_coo_sorted(const size_t *rows, const size_t *cols,
                              size_t nnz) {
  if (!rows || !cols || nnz <= 1) {
    return true;
  }

  for (size_t k = 1; k < nnz; ++k) {
    if (dms_compare_coordinates(rows[k - 1], cols[k - 1], rows[k], cols[k]) >
        0) {
      return false;
    }
  }

  return true;
}

static bool dms_size_t_to_int(size_t value, int *out) {
  if (!out) {
    return false;
  }
  if (value > (size_t)INT_MAX) {
    log_error("Error: sparse matrix size exceeds CSparse integer range.\n");
    return false;
  }
  *out = (int)value;
  return true;
}

static bool dms_linear_find(const DoubleSparseMatrix *matrix, size_t i,
                            size_t j, size_t *position) {
  if (!matrix) {
    return false;
  }

  for (size_t k = 0; k < matrix->nnz; ++k) {
    if (matrix->row_indices[k] == i && matrix->col_indices[k] == j) {
      if (position) {
        *position = k;
      }
      return true;
    }
  }

  return false;
}

// Binary search requires COO triples sorted by (row, col) in row-major order.
// Builder mode may temporarily violate that ordering until dms_ensure_sorted()
// restores the invariant.
static size_t dms_binary_search(const DoubleSparseMatrix *matrix, size_t i,
                                size_t j) {
  size_t low = 0;
  size_t high = matrix->nnz;

  while (low < high) {
    size_t mid = low + ((high - low) >> 1);

    if (matrix->row_indices[mid] == i && matrix->col_indices[mid] == j) {
      return mid;
    }
    if (matrix->row_indices[mid] < i ||
        (matrix->row_indices[mid] == i && matrix->col_indices[mid] < j)) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

static size_t dms_lower_bound_row(const DoubleSparseMatrix *matrix, size_t row) {
  size_t low = 0;
  size_t high = matrix->nnz;

  while (low < high) {
    size_t mid = low + ((high - low) >> 1);
    if (matrix->row_indices[mid] < row) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

static size_t dms_upper_bound_row(const DoubleSparseMatrix *matrix, size_t row) {
  size_t low = 0;
  size_t high = matrix->nnz;

  while (low < high) {
    size_t mid = low + ((high - low) >> 1);
    if (matrix->row_indices[mid] <= row) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

static bool dms_append_element(DoubleSparseMatrix *matrix, size_t i, size_t j,
                               double value, bool keeps_sorted) {
  if (!matrix) {
    return false;
  }

  if (matrix->nnz == matrix->capacity) {
    size_t new_capacity =
        matrix->capacity > 0 ? matrix->capacity * 2 : (size_t)INIT_CAPACITY;
    if (new_capacity <= matrix->capacity) {
      new_capacity = matrix->capacity + (size_t)INIT_CAPACITY;
    }
    if (!dms_realloc(matrix, new_capacity)) {
      return false;
    }
  }

  matrix->row_indices[matrix->nnz] = i;
  matrix->col_indices[matrix->nnz] = j;
  matrix->values[matrix->nnz] = value;
  matrix->nnz++;

  dms_matrix_set_sorted(matrix, keeps_sorted);
  return true;
}

static bool dms_insert_element_sorted(DoubleSparseMatrix *matrix, size_t i,
                                      size_t j, double value,
                                      size_t position) {
  size_t shift = 0;

  if (!matrix) {
    return false;
  }

  if (matrix->nnz == matrix->capacity) {
    size_t new_capacity =
        matrix->capacity > 0 ? matrix->capacity * 2 : (size_t)INIT_CAPACITY;
    if (new_capacity <= matrix->capacity) {
      new_capacity = matrix->capacity + (size_t)INIT_CAPACITY;
    }
    if (!dms_realloc(matrix, new_capacity)) {
      return false;
    }
  }

  shift = matrix->nnz - position;
  if (shift > 0) {
    memmove(&matrix->row_indices[position + 1], &matrix->row_indices[position],
            shift * sizeof(size_t));
    memmove(&matrix->col_indices[position + 1], &matrix->col_indices[position],
            shift * sizeof(size_t));
    memmove(&matrix->values[position + 1], &matrix->values[position],
            shift * sizeof(double));
  }

  matrix->row_indices[position] = i;
  matrix->col_indices[position] = j;
  matrix->values[position] = value;
  matrix->nnz++;

  dms_matrix_set_sorted(matrix, true);
  return true;
}

static int dms_compare_indices(size_t ia, size_t ib,
                               const dms_sort_context *ctx) {
  return dms_compare_coordinates(ctx->rows[ia], ctx->cols[ia], ctx->rows[ib],
                                 ctx->cols[ib]);
}

static void dms_merge_sorted_indices(size_t *idx, size_t *scratch, size_t left,
                                     size_t mid, size_t right,
                                     const dms_sort_context *ctx) {
  size_t i = left;
  size_t j = mid;
  size_t k = left;

  while (i < mid && j < right) {
    if (dms_compare_indices(idx[i], idx[j], ctx) <= 0) {
      scratch[k++] = idx[i++];
    } else {
      scratch[k++] = idx[j++];
    }
  }

  while (i < mid) {
    scratch[k++] = idx[i++];
  }
  while (j < right) {
    scratch[k++] = idx[j++];
  }

  memcpy(&idx[left], &scratch[left], (right - left) * sizeof(size_t));
}

static void dms_sort_indices(size_t *idx, size_t *scratch, size_t left,
                             size_t right, const dms_sort_context *ctx) {
  if (right - left <= 1) {
    return;
  }

  size_t mid = left + ((right - left) >> 1);
  dms_sort_indices(idx, scratch, left, mid, ctx);
  dms_sort_indices(idx, scratch, mid, right, ctx);
  dms_merge_sorted_indices(idx, scratch, left, mid, right, ctx);
}

// Sorting restores the public COO invariant after builder-mode appends and
// after conversions that materialize data in a different traversal order.
static bool dms_sort_coo(DoubleSparseMatrix *mat) {
  size_t *sort_buffers = NULL;
  size_t *tmp_indices = NULL;
  double *tmp_values = NULL;
  size_t *idx = NULL;
  size_t *scratch = NULL;
  size_t *tmp_rows = NULL;
  size_t *tmp_cols = NULL;
  dms_sort_context ctx;

  if (!mat || mat->nnz <= 1) {
    dms_matrix_set_sorted(mat, true);
    return true;
  }

  sort_buffers = malloc(mat->nnz * 2 * sizeof(size_t));
  tmp_indices = malloc(mat->nnz * 2 * sizeof(size_t));
  tmp_values = malloc(mat->nnz * sizeof(double));
  if (!sort_buffers || !tmp_indices || !tmp_values) {
    free(sort_buffers);
    free(tmp_indices);
    free(tmp_values);
    return false;
  }

  idx = sort_buffers;
  scratch = sort_buffers + mat->nnz;
  tmp_rows = tmp_indices;
  tmp_cols = tmp_indices + mat->nnz;

  for (size_t k = 0; k < mat->nnz; ++k) {
    idx[k] = k;
  }

  ctx.rows = mat->row_indices;
  ctx.cols = mat->col_indices;
  dms_sort_indices(idx, scratch, 0, mat->nnz, &ctx);

  for (size_t k = 0; k < mat->nnz; ++k) {
    tmp_rows[k] = mat->row_indices[idx[k]];
    tmp_cols[k] = mat->col_indices[idx[k]];
    tmp_values[k] = mat->values[idx[k]];
  }

  memcpy(mat->row_indices, tmp_rows, mat->nnz * sizeof(size_t));
  memcpy(mat->col_indices, tmp_cols, mat->nnz * sizeof(size_t));
  memcpy(mat->values, tmp_values, mat->nnz * sizeof(double));

  dms_matrix_set_sorted(mat, true);

  free(sort_buffers);
  free(tmp_indices);
  free(tmp_values);
  return true;
}

static bool dms_ensure_sorted(DoubleSparseMatrix *mat) {
  if (!mat) {
    return false;
  }
  if (dms_matrix_is_sorted(mat)) {
    return true;
  }
  if (!dms_sort_coo(mat)) {
    log_error("Error: unable to sort sparse COO triples.\n");
    return false;
  }
  return true;
}

static void dms_build_row_offsets(const DoubleSparseMatrix *mat,
                                  size_t *row_offsets) {
  size_t k = 0;

  row_offsets[0] = 0;
  for (size_t row = 0; row < mat->rows; ++row) {
    while (k < mat->nnz && mat->row_indices[k] == row) {
      ++k;
    }
    row_offsets[row + 1] = k;
  }
}

bool dms_spmv(const DoubleSparseMatrix *mat, const double *x, double *y) {
  size_t *row_offsets = NULL;

  if (!mat || !x || !y) {
    return false;
  }
  if (!dms_ensure_sorted((DoubleSparseMatrix *)mat)) {
    return false;
  }

  row_offsets = calloc(mat->rows + 1, sizeof(size_t));
  if (!row_offsets) {
    return false;
  }

  dms_build_row_offsets(mat, row_offsets);

#pragma omp parallel for if(mat->rows > 64)
  for (size_t row = 0; row < mat->rows; ++row) {
    double sum = 0.0;
    for (size_t k = row_offsets[row]; k < row_offsets[row + 1]; ++k) {
      sum += mat->values[k] * x[mat->col_indices[k]];
    }
    y[row] = sum;
  }

  free(row_offsets);
  return true;
}

static uint64_t dms_mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static uint64_t dms_resolve_seed(uint64_t seed) {
  if (seed != 0) {
    return seed;
  }
  if (dms_seed_initialized) {
    return dms_global_seed;
  }
  return ((uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)&dms_global_seed);
}

void dms_set_random_seed(uint64_t seed) {
  dms_global_seed = seed;
  dms_seed_initialized = true;
}

uint64_t dms_get_random_seed(void) {
  if (!dms_seed_initialized) {
    return 0;
  }
  return dms_global_seed;
}

/*******************************/
/*       Public Functions      */
/*******************************/

DoubleSparseMatrix *dms_create_with_values(size_t rows, size_t cols, size_t nnz,
                                           size_t *row_indices,
                                           size_t *col_indices,
                                           double *values) {
  DoubleSparseMatrix *mat =
      dms_create(rows, cols, nnz > 0 ? nnz : (size_t)1);
  if (!mat) {
    return NULL;
  }

  if (nnz > 0) {
    memcpy(mat->row_indices, row_indices, nnz * sizeof(size_t));
    memcpy(mat->col_indices, col_indices, nnz * sizeof(size_t));
    memcpy(mat->values, values, nnz * sizeof(double));
  }
  mat->nnz = nnz;
  dms_matrix_set_state(mat, dms_is_coo_sorted(row_indices, col_indices, nnz),
                       true);

  return mat;
}

DoubleSparseMatrix *dms_create_empty(void) {
  DoubleSparseMatrix *mat = malloc(sizeof(DoubleSparseMatrix));
  if (!mat) {
    log_error("Error allocating memory for matrix struct");
    return NULL;
  }

  mat->rows = 0;
  mat->cols = 0;
  mat->nnz = 0;
  mat->capacity = 0;
  mat->row_indices = NULL;
  mat->col_indices = NULL;
  mat->values = NULL;

  return mat;
}

DoubleSparseMatrix *dms_create(size_t rows, size_t cols, size_t capacity) {
  DoubleSparseMatrix *mat = NULL;

  if (rows < 1 || cols < 1) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  if (capacity == 0) {
    log_error("Error: matrix capacity cannot be zero.");
    return NULL;
  }

  mat = malloc(sizeof(DoubleSparseMatrix));
  if (!mat) {
    log_error("Error allocating memory for matrix struct");
    return NULL;
  }

  mat->rows = rows;
  mat->cols = cols;
  mat->nnz = 0;
  mat->capacity = capacity;
  mat->row_indices = NULL;
  mat->col_indices = NULL;
  mat->values = NULL;

  if (!dms_allocate_storage(mat, capacity, true, true)) {
    free(mat);
    return NULL;
  }

  return mat;
}

DoubleSparseMatrix *dms_clone(const DoubleSparseMatrix *m) {
  DoubleSparseMatrix *copy = NULL;

  if (!m) {
    return NULL;
  }

  if (!m->row_indices || !m->col_indices || !m->values || m->capacity == 0) {
    copy = dms_create_empty();
    if (!copy) {
      return NULL;
    }
    copy->rows = m->rows;
    copy->cols = m->cols;
    copy->nnz = m->nnz;
    copy->capacity = m->capacity;
    return copy;
  }

  copy = dms_create(m->rows, m->cols, m->capacity);
  if (!copy) {
    return NULL;
  }

  copy->nnz = m->nnz;
  if (m->nnz > 0) {
    memcpy(copy->row_indices, m->row_indices, m->nnz * sizeof(size_t));
    memcpy(copy->col_indices, m->col_indices, m->nnz * sizeof(size_t));
    memcpy(copy->values, m->values, m->nnz * sizeof(double));
  }
  dms_matrix_set_state(copy, dms_matrix_is_sorted(m),
                       dms_matrix_builder_mode(m));

  return copy;
}

DoubleSparseMatrix *dms_create_clone(const DoubleSparseMatrix *m) {
  return dms_clone(m);
}

DoubleSparseMatrix *dms_create_identity(size_t n) {
  DoubleSparseMatrix *mat = NULL;

  if (n < 1) {
    log_error("Error: invalid identity dimensions.\n");
    return NULL;
  }

  mat = dms_create(n, n, n);
  if (!mat) {
    return NULL;
  }

  for (size_t i = 0; i < n; i++) {
    mat->row_indices[i] = i;
    mat->col_indices[i] = i;
    mat->values[i] = 1.0;
  }
  mat->nnz = n;
  dms_matrix_set_state(mat, true, true);

  return mat;
}

cs *dms_to_cs(const DoubleSparseMatrix *coo) {
  DoubleSparseMatrix *mutable_coo = NULL;
  cs *A = NULL;
  int *next = NULL;
  int m = 0;
  int n = 0;
  int nz = 0;

  if (!coo || !coo->row_indices || !coo->col_indices || !coo->values) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }

  // Builder mode may defer ordering; restore sorted row-major COO before
  // materializing the CSC structure expected by CSparse.
  mutable_coo = (DoubleSparseMatrix *)coo;
  if (!dms_ensure_sorted(mutable_coo)) {
    return NULL;
  }

  if (!dms_size_t_to_int(coo->rows, &m) || !dms_size_t_to_int(coo->cols, &n) ||
      !dms_size_t_to_int(coo->nnz, &nz)) {
    return NULL;
  }

  A = cs_spalloc(m, n, nz > 0 ? nz : 1, 1, 0);
  if (!A) {
    return NULL;
  }

  memset(A->p, 0, (size_t)(n + 1) * sizeof(int));

  for (size_t k = 0; k < coo->nnz; ++k) {
    A->p[coo->col_indices[k] + 1]++;
  }
  for (int col = 0; col < n; ++col) {
    A->p[col + 1] += A->p[col];
  }

  next = malloc((size_t)n * sizeof(int));
  if (!next) {
    cs_spfree(A);
    return NULL;
  }
  memcpy(next, A->p, (size_t)n * sizeof(int));

  for (size_t k = 0; k < coo->nnz; ++k) {
    size_t col = coo->col_indices[k];
    int dest = next[col]++;
    A->i[dest] = (int)coo->row_indices[k];
    A->x[dest] = coo->values[k];
  }

  A->nz = -1;

  free(next);
  return A;
}

DoubleSparseMatrix *dms_from_cs(const cs *A) {
  DoubleSparseMatrix *coo = NULL;
  size_t nnz = 0;

  if (!A || !A->p || !A->i || !A->x) {
    return NULL;
  }

  coo = dms_create_empty();
  if (!coo) {
    return NULL;
  }

  coo->rows = (size_t)A->m;
  coo->cols = (size_t)A->n;
  nnz = (size_t)A->p[A->n];
  coo->nnz = 0;
  coo->capacity = nnz + INIT_CAPACITY;
  if (coo->capacity == 0) {
    coo->capacity = 1;
  }

  if (!dms_allocate_storage(coo, coo->capacity, false, true)) {
    free(coo);
    return NULL;
  }

  // CSparse exposes columns contiguously. That traversal is not row-major, so
  // builder mode marks the matrix unsorted until a binary-search path needs it.
  for (int col = 0; col < A->n; ++col) {
    for (size_t p = (size_t)A->p[col]; p < (size_t)A->p[col + 1]; ++p) {
      coo->row_indices[coo->nnz] = (size_t)A->i[p];
      coo->col_indices[coo->nnz] = (size_t)col;
      coo->values[coo->nnz] = A->x[p];
      coo->nnz++;
    }
  }

  dms_matrix_set_state(coo, false, true);
  return coo;
}

DoubleSparseMatrix *cs_to_dms(const cs *A) { return dms_from_cs(A); }

DoubleSparseMatrix *dms_create_random_seeded(size_t rows, size_t cols,
                                             double density, uint64_t seed) {
  DoubleSparseMatrix *mat = NULL;
  uint64_t base_seed = 0;
  double nnz_d = 0.0;
  size_t nnz = 0;

  nnz_d = (double)rows * (double)cols * density;
  nnz = (size_t)nnz_d;
  if (nnz == 0) {
    mat = dms_create(rows, cols, 1);
    if (!mat) {
      return NULL;
    }
    mat->nnz = 0;
    return mat;
  }

  mat = dms_create(rows, cols, nnz);
  if (!mat) {
    return NULL;
  }

  base_seed = dms_resolve_seed(seed);

#pragma omp parallel for
  for (size_t k = 0; k < nnz; ++k) {
    uint64_t row_mix =
        dms_mix64(base_seed ^ ((uint64_t)k * 0x9E3779B97F4A7C15ull));
    uint64_t col_mix =
        dms_mix64(base_seed ^ ((uint64_t)k * 0xD2B74407B1CE6E93ull));
    uint64_t val_mix =
        dms_mix64(base_seed ^ ((uint64_t)k * 0x94D049BB133111EBull));

    mat->row_indices[k] = (size_t)(row_mix % rows);
    mat->col_indices[k] = (size_t)(col_mix % cols);
    mat->values[k] = (double)(val_mix >> 11) / 9007199254740992.0;
  }

  mat->nnz = nnz;

  // Random generation intentionally leaves COO entries unsorted so builder mode
  // can defer the repair cost until a binary-search-based access is needed.
  dms_matrix_set_state(mat, false, true);
  return mat;
}

DoubleSparseMatrix *dms_create_random(size_t rows, size_t cols,
                                      double density) {
  return dms_create_random_seeded(rows, cols, density, 0);
}

DoubleSparseMatrix *dms_create_from_array(size_t rows, size_t cols,
                                          double *array) {
  DoubleSparseMatrix *mat = NULL;
  size_t nnz = 0;
  size_t k = 0;

  if (!array) {
    return NULL;
  }

  for (size_t idx = 0; idx < rows * cols; idx++) {
    if (array[idx] != 0) {
      nnz++;
    }
  }

  mat = dms_create(rows, cols, nnz > 0 ? nnz : 1);
  if (!mat) {
    return NULL;
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (array[i * cols + j] != 0) {
        mat->row_indices[k] = i;
        mat->col_indices[k] = j;
        mat->values[k] = array[i * cols + j];
        k++;
      }
    }
  }
  mat->nnz = k;
  dms_matrix_set_state(mat, true, true);

  return mat;
}

DoubleSparseMatrix *dms_from_array_static(size_t rows, size_t cols,
                                          double array[rows][cols]) {
  DoubleSparseMatrix *mat = NULL;
  size_t nnz = 0;
  size_t k = 0;

  if (!array) {
    return NULL;
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (array[i][j] != 0) {
        nnz++;
      }
    }
  }

  mat = dms_create(rows, cols, nnz > 0 ? nnz : 1);
  if (!mat) {
    return NULL;
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (array[i][j] != 0) {
        mat->row_indices[k] = i;
        mat->col_indices[k] = j;
        mat->values[k] = array[i][j];
        k++;
      }
    }
  }
  mat->nnz = nnz;
  dms_matrix_set_state(mat, true, true);

  return mat;
}

DoubleSparseMatrix *dms_create_from_2D_array(size_t rows, size_t cols,
                                             double array[rows][cols]) {
  return dms_from_array_static(rows, cols, array);
}

DoubleSparseMatrix *dms_get_row(const DoubleSparseMatrix *mat, size_t i) {
  DoubleSparseMatrix *row = NULL;
  size_t start = 0;
  size_t end = 0;
  size_t count = 0;

  if (!mat) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }
  if (i >= mat->rows) {
    log_error("Error: invalid row index.\n");
    return NULL;
  }
  if (!dms_ensure_sorted((DoubleSparseMatrix *)mat)) {
    return NULL;
  }

  start = dms_lower_bound_row(mat, i);
  end = dms_upper_bound_row(mat, i);
  count = end - start;

  row = dms_create(1, mat->cols, count > 0 ? count : 1);
  if (!row) {
    return NULL;
  }

  for (size_t k = 0; k < count; ++k) {
    row->row_indices[k] = 0;
  }
  if (count > 0) {
    memcpy(row->col_indices, mat->col_indices + start, count * sizeof(size_t));
    memcpy(row->values, mat->values + start, count * sizeof(double));
  }
  row->nnz = count;
  dms_matrix_set_state(row, true, true);

  return row;
}

DoubleSparseMatrix *dms_get_last_row(const DoubleSparseMatrix *mat) {
  if (!mat || mat->rows == 0) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }
  return dms_get_row(mat, mat->rows - 1);
}

DoubleSparseMatrix *dms_get_col(const DoubleSparseMatrix *mat, size_t j) {
  DoubleSparseMatrix *col = NULL;
  size_t count = 0;
  size_t out = 0;

  if (!mat) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }
  if (j >= mat->cols) {
    log_error("Error: invalid column index.\n");
    return NULL;
  }
  if (!dms_ensure_sorted((DoubleSparseMatrix *)mat)) {
    return NULL;
  }

  for (size_t k = 0; k < mat->nnz; ++k) {
    if (mat->col_indices[k] == j) {
      count++;
    }
  }

  col = dms_create(mat->rows, 1, count > 0 ? count : 1);
  if (!col) {
    return NULL;
  }

  for (size_t k = 0; k < mat->nnz; ++k) {
    if (mat->col_indices[k] == j) {
      col->row_indices[out] = mat->row_indices[k];
      col->col_indices[out] = 0;
      col->values[out] = mat->values[k];
      out++;
    }
  }
  col->nnz = out;
  dms_matrix_set_state(col, true, true);

  return col;
}

DoubleSparseMatrix *dms_get_last_col(const DoubleSparseMatrix *mat) {
  if (!mat || mat->cols == 0) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }
  return dms_get_col(mat, mat->cols - 1);
}

DoubleSparseMatrix *dms_multiply(const DoubleSparseMatrix *mat1,
                                 const DoubleSparseMatrix *mat2) {
  DoubleSparseMatrix *result = NULL;
  cs *A = NULL;
  cs *B = NULL;
  cs *C = NULL;

  if (!mat1 || !mat2) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }
  if (mat1->cols != mat2->rows) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }

  A = dms_to_cs(mat1);
  B = dms_to_cs(mat2);
  if (!A || !B) {
    cs_spfree(A);
    cs_spfree(B);
    return NULL;
  }

  C = cs_multiply(A, B);
  result = C ? dms_from_cs(C) : NULL;

  cs_spfree(A);
  cs_spfree(B);
  cs_spfree(C);

  return result;
}

DoubleSparseMatrix *dms_multiply_by_number(const DoubleSparseMatrix *mat,
                                           const double number) {
  DoubleSparseMatrix *result = NULL;

  if (!mat) {
    return NULL;
  }

  result = dms_clone(mat);
  if (!result) {
    return NULL;
  }

  for (size_t i = 0; i < mat->nnz; i++) {
    result->values[i] *= number;
  }

  return result;
}

DoubleSparseMatrix *dms_transpose(const DoubleSparseMatrix *mat) {
  DoubleSparseMatrix *result = NULL;
  cs *A = NULL;
  cs *AT = NULL;

  if (!mat) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }
  if (mat->col_indices == NULL || mat->row_indices == NULL ||
      mat->values == NULL) {
    log_error("Error: matrix is empty.\n");
    return NULL;
  }
  if (mat->nnz == 0) {
    return dms_create(mat->cols, mat->rows, 1);
  }

  A = dms_to_cs(mat);
  if (!A) {
    return NULL;
  }

  AT = cs_transpose(A, 1);
  result = AT ? dms_from_cs(AT) : NULL;

  cs_spfree(A);
  cs_spfree(AT);

  return result;
}

double dms_get(const DoubleSparseMatrix *mat, size_t i, size_t j) {
  size_t pos = 0;

  if (!mat) {
    log_error("Error: invalid sparse matrix input.\n");
    return 0.0;
  }
  if (i >= mat->rows || j >= mat->cols) {
    log_error("Error: matrix index out of bounds.\n");
    return 0.0;
  }

  if (!dms_ensure_sorted((DoubleSparseMatrix *)mat)) {
    return 0.0;
  }

  pos = dms_binary_search(mat, i, j);
  if (pos < mat->nnz && mat->row_indices[pos] == i &&
      mat->col_indices[pos] == j) {
    return mat->values[pos];
  }

  return 0.0;
}

bool dms_set(DoubleSparseMatrix *matrix, size_t i, size_t j, double value) {
  if (!matrix || !matrix->row_indices || !matrix->col_indices ||
      !matrix->values) {
    log_error("Error: invalid sparse matrix input.\n");
    return false;
  }
  if (i >= matrix->rows || j >= matrix->cols) {
    log_error("Error: matrix index out of bounds.\n");
    return false;
  }

  if (dms_matrix_builder_mode(matrix)) {
    if (matrix->nnz == 0) {
      return dms_append_element(matrix, i, j, value, true);
    }

    if (dms_matrix_is_sorted(matrix)) {
      size_t last = matrix->nnz - 1;
      int tail_cmp = dms_compare_coordinates(
          matrix->row_indices[last], matrix->col_indices[last], i, j);

      if (tail_cmp < 0) {
        return dms_append_element(matrix, i, j, value, true);
      }
      if (tail_cmp == 0) {
        matrix->values[last] = value;
        return true;
      }

      size_t position = dms_binary_search(matrix, i, j);
      if (position < matrix->nnz && matrix->row_indices[position] == i &&
          matrix->col_indices[position] == j) {
        matrix->values[position] = value;
        return true;
      }

      // Builder mode avoids O(nnz) memmove here by appending out-of-order data
      // and marking the matrix unsorted until a later binary-search path.
      return dms_append_element(matrix, i, j, value, false);
    }

    {
      size_t position = 0;
      if (dms_linear_find(matrix, i, j, &position)) {
        matrix->values[position] = value;
        return true;
      }
    }
    return dms_append_element(matrix, i, j, value, false);
  }

  if (!dms_ensure_sorted(matrix)) {
    return false;
  }

  {
    size_t position = dms_binary_search(matrix, i, j);
    if (position < matrix->nnz && matrix->row_indices[position] == i &&
        matrix->col_indices[position] == j) {
      matrix->values[position] = value;
      return true;
    }
    return dms_insert_element_sorted(matrix, i, j, value, position);
  }
}

bool dms_realloc(DoubleSparseMatrix *mat, size_t new_capacity) {
  dms_index_storage *new_indices = NULL;
  double *new_values = NULL;
  bool is_sorted = false;
  bool builder_mode = false;

  if (!mat) {
    log_error("Error: invalid sparse matrix input.\n");
    return false;
  }
  if (new_capacity <= mat->capacity) {
    log_error("Error: cannot resize matrix to smaller/equal capacity.\n");
    return false;
  }
  if (!mat->row_indices || !mat->col_indices || !mat->values) {
    log_error("Error: sparse matrix storage is not initialized.\n");
    return false;
  }

  is_sorted = dms_matrix_is_sorted(mat);
  builder_mode = dms_matrix_builder_mode(mat);

  new_indices =
      malloc(sizeof(dms_index_storage) + (new_capacity * 2 * sizeof(size_t)));
  if (!new_indices) {
    log_error("Error allocating row indices for sparse matrix resize!\n");
    return false;
  }

  new_values = malloc(new_capacity * sizeof(double));
  if (!new_values) {
    log_error("Error allocating values for sparse matrix resize!\n");
    free(new_indices);
    return false;
  }

  new_indices->is_sorted = is_sorted;
  new_indices->builder_mode = builder_mode;
  new_indices->reserved = 0;

  if (mat->nnz > 0) {
    memcpy(new_indices->data, mat->row_indices, mat->nnz * sizeof(size_t));
    memcpy(new_indices->data + new_capacity, mat->col_indices,
           mat->nnz * sizeof(size_t));
    memcpy(new_values, mat->values, mat->nnz * sizeof(double));
  }

  free(dms_index_storage_from_rows(mat->row_indices));
  free(mat->values);

  mat->row_indices = new_indices->data;
  mat->col_indices = new_indices->data + new_capacity;
  mat->values = new_values;
  mat->capacity = new_capacity;

  return true;
}

void dms_print(const DoubleSparseMatrix *mat) {
  if (!mat) {
    return;
  }
  if (mat->cols > 100 || mat->rows > 100) {
    printf("Matrix is too large to print, showing COO triples:\n");
    printf("Matrix: %zu x %zu, nnz: %zu\n", mat->rows, mat->cols, mat->nnz);
    for (size_t i = 0; i < mat->nnz && i < 20; i++) {
      printf("  (%zu, %zu) = %.4lf\n", mat->row_indices[i],
             mat->col_indices[i], mat->values[i]);
    }
    if (mat->nnz > 20) {
      printf("  ... (%zu more entries)\n", mat->nnz - 20);
    }
    return;
  }
  if (mat->nnz == 0) {
    printf("Empty matrix\n");
    return;
  }

  printf("Matrix: %zu x %zu\n", mat->rows, mat->cols);
  {
    double *dense = dms_to_array(mat);
    if (!dense) {
      return;
    }
    for (size_t i = 0; i < mat->rows; i++) {
      for (size_t j = 0; j < mat->cols; j++) {
        printf("%.2lf ", dense[i * mat->cols + j]);
      }
      printf("\n");
    }
    free(dense);
  }
}

void dms_destroy(DoubleSparseMatrix *mat) {
  if (!mat) {
    return;
  }
  dms_free_storage(mat);
  free(mat);
}

double *dms_to_array(const DoubleSparseMatrix *mat) {
  double *array = NULL;
  size_t n = 0;

  if (!mat) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }

  n = mat->rows * mat->cols;
  array = calloc(n, sizeof(double));
  if (!array) {
    log_error("Error: unable to allocate memory for array.\n");
    return NULL;
  }

  for (size_t k = 0; k < mat->nnz; k++) {
    array[mat->row_indices[k] * mat->cols + mat->col_indices[k]] =
        mat->values[k];
  }

  return array;
}

double dms_density(const DoubleSparseMatrix *mat) {
  if (!mat || mat->rows == 0 || mat->cols == 0) {
    return 0.0;
  }
  return (double)mat->nnz / (double)(mat->rows * mat->cols);
}

#endif  // DMS_HAS_FULL_DEPS
