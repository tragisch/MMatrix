/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "dms.h"

#if defined(__has_include)
#if __has_include(<cs.h>) && __has_include(<omp.h>)
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

static void dms_invalidate_csc_cache(DoubleSparseMatrix *mat) {
  if (!mat) {
    return;
  }
  if (mat->csc_cache) {
    cs_spfree(mat->csc_cache);
    mat->csc_cache = NULL;
  }
  mat->csc_valid = false;
  /* Also invalidate the CSR (row_offsets) cache. */
  free(mat->csr_cache);
  mat->csr_cache = NULL;
  mat->csr_valid = false;
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

  dms_invalidate_csc_cache(matrix);
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

  dms_invalidate_csc_cache(matrix);
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
  char *workspace = NULL;
  size_t *idx = NULL;
  size_t *scratch = NULL;
  size_t *tmp_rows = NULL;
  size_t *tmp_cols = NULL;
  double *tmp_values = NULL;
  dms_sort_context ctx;

  if (!mat || mat->nnz <= 1) {
    dms_matrix_set_sorted(mat, true);
    return true;
  }

  /* Single allocation for all temporary sort buffers. */
  {
    size_t idx_bytes = mat->nnz * 4 * sizeof(size_t);
    size_t val_bytes = mat->nnz * sizeof(double);
    workspace = malloc(idx_bytes + val_bytes);
    if (!workspace) {
      return false;
    }
  }

  idx = (size_t *)workspace;
  scratch = idx + mat->nnz;
  tmp_rows = scratch + mat->nnz;
  tmp_cols = tmp_rows + mat->nnz;
  tmp_values = (double *)(tmp_cols + mat->nnz);

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

  free(workspace);

  /* Deduplicate: for consecutive entries sharing the same (row, col),
     keep only the last one (last-write-wins; the stable merge sort
     preserves insertion order among equal keys). */
  {
    size_t write = 0;
    for (size_t k = 0; k < mat->nnz; ++k) {
      if (k + 1 < mat->nnz &&
          mat->row_indices[k] == mat->row_indices[k + 1] &&
          mat->col_indices[k] == mat->col_indices[k + 1]) {
        continue;
      }
      if (write != k) {
        mat->row_indices[write] = mat->row_indices[k];
        mat->col_indices[write] = mat->col_indices[k];
        mat->values[write] = mat->values[k];
      }
      write++;
    }
    mat->nnz = write;
  }

  dms_matrix_set_sorted(mat, true);
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

// COO remains the builder/storage format. This helper materializes and retains
// the CSC compute representation lazily, then reuses it until a mutation
// invalidates the cache.
static cs *dms_get_csc(DoubleSparseMatrix *mat) {
  if (!mat) {
    return NULL;
  }
  if (mat->csc_valid && mat->csc_cache) {
    return mat->csc_cache;
  }

  dms_invalidate_csc_cache(mat);
  mat->csc_cache = dms_to_cs(mat);
  if (!mat->csc_cache) {
    return NULL;
  }
  mat->csc_valid = true;
  return mat->csc_cache;
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

/* Lazily compute and cache CSR row_offsets.  The matrix must be sorted
   (row-major) before calling this.  Returns the cached array or NULL on
   allocation failure.  Callers must NOT free the returned pointer. */
static const size_t *dms_get_csr(DoubleSparseMatrix *mat) {
  if (!mat) return NULL;
  if (mat->csr_valid && mat->csr_cache) return mat->csr_cache;

  size_t *rp = calloc(mat->rows + 1, sizeof(size_t));
  if (!rp) return NULL;
  dms_build_row_offsets(mat, rp);

  /* Replace any stale cache. */
  free(mat->csr_cache);
  mat->csr_cache = rp;
  mat->csr_valid = true;
  return rp;
}
bool dms_spmv(const DoubleSparseMatrix *mat, const double *x, double *y) {
  if (!mat || !x || !y) {
    return false;
  }
  if (!dms_ensure_sorted((DoubleSparseMatrix *)mat)) {
    return false;
  }

  /* Fast path for small/sparse matrices: direct scatter avoids the
     row_offsets heap allocation. */
  if (mat->rows <= 4096 || mat->nnz < 8192) {
    memset(y, 0, mat->rows * sizeof(double));
    for (size_t k = 0; k < mat->nnz; ++k) {
      y[mat->row_indices[k]] += mat->values[k] * x[mat->col_indices[k]];
    }
    return true;
  }

  {
    const size_t *row_offsets = dms_get_csr((DoubleSparseMatrix *)mat);
    if (!row_offsets) {
      return false;
    }

#pragma omp parallel for
    for (size_t row = 0; row < mat->rows; ++row) {
      double sum = 0.0;
      for (size_t k = row_offsets[row]; k < row_offsets[row + 1]; ++k) {
        sum += mat->values[k] * x[mat->col_indices[k]];
      }
      y[row] = sum;
    }
  }
  return true;
}

/* ------------------------------------------------------------------ */
/*  Native COO transpose — avoids the COO→CSC→transpose→CSC→COO      */
/*  round-trip through CSparse.  O(nnz + rows) with a single pass.    */
/* ------------------------------------------------------------------ */
static DoubleSparseMatrix *dms_transpose_native(const DoubleSparseMatrix *mat) {
  DoubleSparseMatrix *out = NULL;

  if (!mat) {
    return NULL;
  }
  if (!mat->row_indices || !mat->col_indices || !mat->values) {
    return NULL;
  }
  if (mat->nnz == 0) {
    return dms_create(mat->cols, mat->rows, 1);
  }
  if (!dms_ensure_sorted((DoubleSparseMatrix *)mat)) {
    return NULL;
  }

  out = dms_create(mat->cols, mat->rows, mat->nnz);
  if (!out) {
    return NULL;
  }

  /* Counting-sort by column (which becomes the new row). */
  {
    size_t *counts = calloc(mat->cols + 1, sizeof(size_t));
    if (!counts) {
      dms_destroy(out);
      return NULL;
    }

    for (size_t k = 0; k < mat->nnz; ++k) {
      counts[mat->col_indices[k] + 1]++;
    }
    for (size_t c = 0; c < mat->cols; ++c) {
      counts[c + 1] += counts[c];
    }

    for (size_t k = 0; k < mat->nnz; ++k) {
      size_t dest = counts[mat->col_indices[k]]++;
      out->row_indices[dest] = mat->col_indices[k];
      out->col_indices[dest] = mat->row_indices[k];
      out->values[dest] = mat->values[k];
    }

    free(counts);
  }

  out->nnz = mat->nnz;
  dms_matrix_set_state(out, true, true);
  return out;
}

/* ------------------------------------------------------------------ */
/*  Native SpGEMM  C = A * B  —  Gustavson row-by-row CSR×CSR         */
/*                                                                     */
/*  Small matrices  → serial single-pass  (low overhead)               */
/*  Large matrices  → parallel single-pass with OpenMP:                */
/*    Each thread gets a contiguous row chunk (schedule(static)) and   */
/*    writes into a thread-local buffer.  Buffers are concatenated     */
/*    after the parallel region — no inter-thread sync needed.         */
/*                                                                     */
/*  Both paths prune numerical zeros and use a hybrid gather strategy  */
/*  (dense scan vs. shell sort) depending on n and per-row fill.       */
/*  Marker arrays use int tags → rows capped at INT_MAX.               */
/* ------------------------------------------------------------------ */

/* Helper: reallocate COO triple arrays (row, col, val) to new_cap.
   Unlike the public dms_realloc this works on raw pointers so we can
   use it for intermediate buffers that aren't yet inside a DMS.       */
static bool dms_grow_coo(size_t **rows_p, size_t **cols_p, double **vals_p,
                         size_t nnz, size_t *cap, size_t need) {
  size_t new_cap = *cap;
  while (new_cap < need) {
    new_cap = new_cap > 0 ? new_cap * 2 : 256;
  }

  /* Use malloc + memcpy + free instead of realloc so that all three
     original pointers remain valid on partial allocation failure. */
  size_t *nr = malloc(new_cap * sizeof(size_t));
  size_t *nc = malloc(new_cap * sizeof(size_t));
  double *nv = malloc(new_cap * sizeof(double));

  if (!nr || !nc || !nv) {
    free(nr);
    free(nc);
    free(nv);
    return false;
  }

  if (nnz > 0) {
    memcpy(nr, *rows_p, nnz * sizeof(size_t));
    memcpy(nc, *cols_p, nnz * sizeof(size_t));
    memcpy(nv, *vals_p, nnz * sizeof(double));
  }

  free(*rows_p);
  free(*cols_p);
  free(*vals_p);

  *rows_p = nr;
  *cols_p = nc;
  *vals_p = nv;
  *cap = new_cap;
  return true;
}

/* -------- serial single-pass (small matrices) --------------------- */
static DoubleSparseMatrix *dms_multiply_serial(
    const DoubleSparseMatrix *A, const DoubleSparseMatrix *B,
    const size_t *rA, const size_t *rB, size_t m, size_t n) {

  size_t k_B = B->rows > 0 ? B->rows : 1;
  size_t avg_B_row = B->nnz / k_B;
  size_t est = A->nnz * (avg_B_row > 0 ? avg_B_row : 1);
  if (est > m * n) est = m * n;
  if (est == 0) est = 1;

  double *acc      = calloc(n, sizeof(double));
  int    *marker   = malloc(n * sizeof(int));
  size_t *col_list = malloc(n * sizeof(size_t));
  size_t *out_rows = malloc(est * sizeof(size_t));
  size_t *out_cols = malloc(est * sizeof(size_t));
  double *out_vals = malloc(est * sizeof(double));
  size_t  out_cap  = est, out_nnz = 0;

  if (!acc || !marker || !col_list ||
      !out_rows || !out_cols || !out_vals) {
    free(acc); free(marker); free(col_list);
    free(out_rows); free(out_cols); free(out_vals);
    return NULL;
  }
  memset(marker, -1, n * sizeof(int));

  for (size_t i = 0; i < m; ++i) {
    size_t row_cnt = 0;

    for (size_t pa = rA[i]; pa < rA[i + 1]; ++pa) {
      double a_val = A->values[pa];
      size_t ka    = A->col_indices[pa];
      for (size_t pb = rB[ka]; pb < rB[ka + 1]; ++pb) {
        size_t jb = B->col_indices[pb];
        acc[jb] += a_val * B->values[pb];
        if (marker[jb] != (int)i) {
          marker[jb] = (int)i;
          col_list[row_cnt++] = jb;
        }
      }
    }
    if (row_cnt == 0) continue;

    if (out_nnz + row_cnt > out_cap) {
      if (!dms_grow_coo(&out_rows, &out_cols, &out_vals,
                        out_nnz, &out_cap, out_nnz + row_cnt)) {
        free(acc); free(marker); free(col_list);
        free(out_rows); free(out_cols); free(out_vals);
        return NULL;
      }
    }

    /* Hybrid gather: dense scan for heavy rows, shell sort for sparse.
       Both paths prune numerical zeros (cancellation artifacts). */
    if (row_cnt > n / 8) {
      for (size_t j = 0; j < n; ++j) {
        if (marker[j] == (int)i) {
          double v = acc[j];
          acc[j] = 0.0;
          if (v != 0.0) {
            out_rows[out_nnz] = i;
            out_cols[out_nnz] = j;
            out_vals[out_nnz] = v;
            out_nnz++;
          }
        }
      }
    } else {
      for (size_t gap = row_cnt / 2; gap > 0; gap /= 2)
        for (size_t a = gap; a < row_cnt; ++a) {
          size_t key = col_list[a]; size_t b = a;
          while (b >= gap && col_list[b - gap] > key) {
            col_list[b] = col_list[b - gap]; b -= gap;
          }
          col_list[b] = key;
        }
      for (size_t c = 0; c < row_cnt; ++c) {
        size_t jb = col_list[c];
        double v = acc[jb];
        acc[jb] = 0.0;
        if (v != 0.0) {
          out_rows[out_nnz] = i;
          out_cols[out_nnz] = jb;
          out_vals[out_nnz] = v;
          out_nnz++;
        }
      }
    }
  }

  free(acc); free(marker); free(col_list);

  DoubleSparseMatrix *R = dms_create(m, n, out_nnz > 0 ? out_nnz : 1);
  if (!R) { free(out_rows); free(out_cols); free(out_vals); return NULL; }
  if (out_nnz > 0) {
    memcpy(R->row_indices, out_rows, out_nnz * sizeof(size_t));
    memcpy(R->col_indices, out_cols, out_nnz * sizeof(size_t));
    memcpy(R->values,      out_vals, out_nnz * sizeof(double));
  }
  R->nnz = out_nnz;
  dms_matrix_set_state(R, true, true);
  free(out_rows); free(out_cols); free(out_vals);
  return R;
}

/* -------- parallel single-pass (large matrices) ------------------- */
/*  Each thread processes a contiguous chunk of rows (schedule(static))
    into a thread-local buffer.  After the parallel region the per-thread
    buffers are concatenated in order — guaranteeing row-sorted output
    without any inter-thread synchronization during computation.        */
static DoubleSparseMatrix *dms_multiply_parallel(
    const DoubleSparseMatrix *A, const DoubleSparseMatrix *B,
    const size_t *rA, const size_t *rB, size_t m, size_t n) {

  int nthreads = 1;
  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  /* Per-thread output buffers. */
  size_t  nt = (size_t)nthreads;
  size_t  *thr_nnz = calloc(nt, sizeof(size_t));
  size_t  *thr_cap = malloc(nt * sizeof(size_t));
  size_t **thr_row = calloc(nt, sizeof(size_t *));
  size_t **thr_col = calloc(nt, sizeof(size_t *));
  double **thr_val = calloc(nt, sizeof(double *));
  if (!thr_nnz || !thr_cap || !thr_row || !thr_col || !thr_val) {
    free(thr_nnz); free(thr_cap); free(thr_row);
    free(thr_col); free(thr_val);
    return NULL;
  }

  /* Estimate: nnz_C ≈ nnz_A × avg_nnz_per_row_B (using B->rows). */
  size_t k_B = B->rows > 0 ? B->rows : 1;
  size_t avg_B_row = B->nnz / k_B;
  size_t est_total = A->nnz * (avg_B_row > 0 ? avg_B_row : 1);
  if (est_total > m * n) est_total = m * n;
  size_t est_per = est_total / nt + 256;
  for (size_t t = 0; t < nt; ++t) {
    thr_cap[t] = est_per;
    thr_row[t] = malloc(est_per * sizeof(size_t));
    thr_col[t] = malloc(est_per * sizeof(size_t));
    thr_val[t] = malloc(est_per * sizeof(double));
  }

  /* Thread-local error flag — each thread has its own copy to avoid
     data races.  Merged after the parallel region. */
  bool *thr_ok = malloc(nt * sizeof(bool));
  if (!thr_ok) {
    for (size_t t = 0; t < nt; ++t) {
      free(thr_row[t]); free(thr_col[t]); free(thr_val[t]);
    }
    free(thr_nnz); free(thr_cap); free(thr_row);
    free(thr_col); free(thr_val);
    return NULL;
  }
  for (size_t t = 0; t < nt; ++t) thr_ok[t] = true;

  /* Decide gather strategy: dense scan for n ≤ threshold,
     hybrid shell-sort / dense scan for larger n. */
  const size_t DENSE_SCAN_LIMIT = 8192; /* ~32 KB per marker array */

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    /* Per-thread work arrays: acc[n] (zero-init) + marker[n] (-1 init)
       + col_list[n] (only when hybrid gather needed).
       Allocated per-thread for proper first-touch on cache lines. */
    bool need_col_list = (n > DENSE_SCAN_LIMIT);
    size_t work_sz = n * sizeof(double) + n * sizeof(int)
                   + (need_col_list ? n * sizeof(size_t) : 0);
    char *work = (char *)malloc(work_sz);
    double *local_acc = NULL;
    int    *local_mk  = NULL;
    size_t *local_cl  = NULL;
    if (!work || !thr_row[tid] || !thr_col[tid] || !thr_val[tid]) {
      thr_ok[tid] = false;
    } else {
      local_acc = (double *)work;
      local_mk  = (int *)(work + n * sizeof(double));
      if (need_col_list)
        local_cl = (size_t *)(work + n * sizeof(double) + n * sizeof(int));
      memset(local_acc, 0, n * sizeof(double));
      memset(local_mk, -1, n * sizeof(int));
    }

    size_t lnnz = 0;

    /* static schedule → thread t gets contiguous rows → row order. */
    #pragma omp for schedule(static)
    for (size_t i = 0; i < m; ++i) {
      if (!thr_ok[tid]) continue;
      size_t row_cnt = 0;

      /* Scatter */
      for (size_t pa = rA[i]; pa < rA[i + 1]; ++pa) {
        double a_val = A->values[pa];
        size_t ka    = A->col_indices[pa];
        for (size_t pb = rB[ka]; pb < rB[ka + 1]; ++pb) {
          size_t jb = B->col_indices[pb];
          local_acc[jb] += a_val * B->values[pb];
          if (local_mk[jb] != (int)i) {
            local_mk[jb] = (int)i;
            if (local_cl) local_cl[row_cnt] = jb;
            row_cnt++;
          }
        }
      }
      if (row_cnt == 0) continue;

      /* Ensure thread-local buffer has room. */
      if (lnnz + row_cnt > thr_cap[tid]) {
        if (!dms_grow_coo(&thr_row[tid], &thr_col[tid], &thr_val[tid],
                          lnnz, &thr_cap[tid], lnnz + row_cnt)) {
          thr_ok[tid] = false; continue;
        }
      }

      /* Gather with zero-pruning.
         Strategy depends on n and row fill ratio:
         - Small n (≤ DENSE_SCAN_LIMIT): always dense scan O(n).
         - Large n, heavy row (> n/8): dense scan O(n).
         - Large n, sparse row: shell-sort col_list O(k log k). */
      if (!local_cl || row_cnt > n / 8) {
        /* Dense scan: produces sorted output naturally. */
        size_t pos = 0;
        for (size_t j = 0; j < n; ++j) {
          if (local_mk[j] == (int)i) {
            double v = local_acc[j];
            local_acc[j] = 0.0;
            if (v != 0.0) {
              thr_row[tid][lnnz + pos] = i;
              thr_col[tid][lnnz + pos] = j;
              thr_val[tid][lnnz + pos] = v;
              pos++;
            }
          }
        }
        lnnz += pos;
      } else {
        /* Shell sort col_list, then sparse emit. */
        for (size_t gap = row_cnt / 2; gap > 0; gap /= 2)
          for (size_t a = gap; a < row_cnt; ++a) {
            size_t key = local_cl[a]; size_t b = a;
            while (b >= gap && local_cl[b - gap] > key) {
              local_cl[b] = local_cl[b - gap]; b -= gap;
            }
            local_cl[b] = key;
          }
        size_t pos = 0;
        for (size_t c = 0; c < row_cnt; ++c) {
          size_t jb = local_cl[c];
          double v = local_acc[jb];
          local_acc[jb] = 0.0;
          if (v != 0.0) {
            thr_row[tid][lnnz + pos] = i;
            thr_col[tid][lnnz + pos] = jb;
            thr_val[tid][lnnz + pos] = v;
            pos++;
          }
        }
        lnnz += pos;
      }
    }

    thr_nnz[tid] = lnnz;
    free(work);
  }

  /* Check if any thread failed. */
  bool any_fail = false;
  for (size_t t = 0; t < nt; ++t) {
    if (!thr_ok[t]) { any_fail = true; break; }
  }
  free(thr_ok);

  if (any_fail) {
    for (int t = 0; t < nthreads; ++t) {
      free(thr_row[t]); free(thr_col[t]); free(thr_val[t]);
    }
    free(thr_nnz); free(thr_cap); free(thr_row);
    free(thr_col); free(thr_val);
    return NULL;
  }

  /* Prefix sum for concatenation offsets. */
  size_t total_nnz = 0;
  for (int t = 0; t < nthreads; ++t) total_nnz += thr_nnz[t];

  /* Build result directly — no intermediate copy when possible. */
  DoubleSparseMatrix *R = dms_create(m, n, total_nnz > 0 ? total_nnz : 1);
  if (R && total_nnz > 0) {
    size_t off = 0;
    for (int t = 0; t < nthreads; ++t) {
      if (thr_nnz[t] > 0) {
        memcpy(R->row_indices + off, thr_row[t], thr_nnz[t] * sizeof(size_t));
        memcpy(R->col_indices + off, thr_col[t], thr_nnz[t] * sizeof(size_t));
        memcpy(R->values      + off, thr_val[t], thr_nnz[t] * sizeof(double));
        off += thr_nnz[t];
      }
    }
    R->nnz = total_nnz;
    dms_matrix_set_state(R, true, true);
  } else if (R) {
    R->nnz = 0;
    dms_matrix_set_state(R, true, true);
  }

  for (int t = 0; t < nthreads; ++t) {
    free(thr_row[t]); free(thr_col[t]); free(thr_val[t]);
  }
  free(thr_nnz); free(thr_cap); free(thr_row);
  free(thr_col); free(thr_val);
  return R;
}

/* -------- dispatch ------------------------------------------------ */
static DoubleSparseMatrix *dms_multiply_native(const DoubleSparseMatrix *mat1,
                                               const DoubleSparseMatrix *mat2) {
  if (!mat1 || !mat2) return NULL;
  if (mat1->cols != mat2->rows) return NULL;

  size_t m = mat1->rows;
  size_t n = mat2->cols;

  /* The marker array uses int to tag rows → row indices must fit in int.
     For matrices with > INT_MAX rows the marker trick would produce
     collisions and wrong results.  Reject early. */
  if (m > (size_t)INT_MAX) return NULL;

  /* Ensure both matrices are sorted (row-major COO ≈ CSR). */
  if (!dms_ensure_sorted((DoubleSparseMatrix *)mat1)) return NULL;
  if (!dms_ensure_sorted((DoubleSparseMatrix *)mat2)) return NULL;

  /* Lazy CSR row pointers for A and B (cached inside the matrix). */
  const size_t *rA = dms_get_csr((DoubleSparseMatrix *)mat1);
  const size_t *rB = dms_get_csr((DoubleSparseMatrix *)mat2);
  if (!rA || !rB) return NULL;

  /* Heuristic: use parallel path for large matrices. */
  size_t total_nnz = mat1->nnz + mat2->nnz;
  if (m >= 128 && total_nnz >= 4096) {
    return dms_multiply_parallel(mat1, mat2, rA, rB, m, n);
  }
  return dms_multiply_serial(mat1, mat2, rA, rB, m, n);
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
  mat->csc_cache = NULL;
  mat->csc_valid = false;
  mat->csr_cache = NULL;
  mat->csr_valid = false;

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
  mat->csc_cache = NULL;
  mat->csc_valid = false;
  mat->csr_cache = NULL;
  mat->csr_valid = false;

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
  copy->csc_cache = NULL;
  copy->csc_valid = false;
  copy->csr_cache = NULL;
  copy->csr_valid = false;
  dms_matrix_set_state(copy, dms_matrix_is_sorted(m),
                       dms_matrix_builder_mode(m));

  return copy;
}

DoubleSparseMatrix *dms_create_clone(const DoubleSparseMatrix *m) {
  return dms_clone(m);
}

DoubleSparseMatrix *dms_create_identity(size_t n) {
  DoubleSparseMatrix *mat = NULL;

  if (n == 0) {
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
  dms_invalidate_csc_cache(mat);
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
  cs *A = NULL;
  size_t count = 0;

  if (!mat) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }
  if (j >= mat->cols) {
    log_error("Error: invalid column index.\n");
    return NULL;
  }

  /* Use the CSC cache for O(nnz_col) column extraction instead of
     scanning all nnz entries. */
  A = dms_get_csc((DoubleSparseMatrix *)mat);
  if (!A) {
    return NULL;
  }

  count = (size_t)(A->p[j + 1] - A->p[j]);

  col = dms_create(mat->rows, 1, count > 0 ? count : 1);
  if (!col) {
    return NULL;
  }

  for (size_t k = 0; k < count; ++k) {
    size_t pos = (size_t)A->p[j] + k;
    col->row_indices[k] = (size_t)A->i[pos];
    col->col_indices[k] = 0;
    col->values[k] = A->x[pos];
  }
  col->nnz = count;
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
  if (!mat1 || !mat2) {
    log_error("Error: invalid sparse matrix input.\n");
    return NULL;
  }
  if (mat1->cols != mat2->rows) {
    log_error("Error: invalid matrix dimensions.\n");
    return NULL;
  }

  return dms_multiply_native(mat1, mat2);
}

DoubleSparseMatrix *dms_multiply_by_number(const DoubleSparseMatrix *mat,
                                           const double number) {
  DoubleSparseMatrix *result = NULL;

  if (!mat) {
    return NULL;
  }

  /* Edge case: uninitialised storage. */
  if (!mat->row_indices || !mat->col_indices || !mat->values ||
      mat->capacity == 0) {
    return dms_clone(mat);
  }

  /* Fused copy + scale: allocate once, copy indices, multiply values
     in a single pass instead of clone + separate scaling loop. */
  result = dms_create(mat->rows, mat->cols, mat->capacity);
  if (!result) {
    return NULL;
  }

  result->nnz = mat->nnz;
  if (mat->nnz > 0) {
    memcpy(result->row_indices, mat->row_indices, mat->nnz * sizeof(size_t));
    memcpy(result->col_indices, mat->col_indices, mat->nnz * sizeof(size_t));
    for (size_t i = 0; i < mat->nnz; i++) {
      result->values[i] = mat->values[i] * number;
    }
  }
  dms_matrix_set_state(result, dms_matrix_is_sorted(mat),
                       dms_matrix_builder_mode(mat));

  return result;
}

DoubleSparseMatrix *dms_transpose(const DoubleSparseMatrix *mat) {
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

  return dms_transpose_native(mat);
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
        dms_invalidate_csc_cache(matrix);
        return true;
      }

      size_t position = dms_binary_search(matrix, i, j);
      if (position < matrix->nnz && matrix->row_indices[position] == i &&
          matrix->col_indices[position] == j) {
        matrix->values[position] = value;
        dms_invalidate_csc_cache(matrix);
        return true;
      }

      // Builder mode avoids O(nnz) memmove here by appending out-of-order data
      // and marking the matrix unsorted until a later binary-search path.
      return dms_append_element(matrix, i, j, value, false);
    }

    // Builder mode (unsorted): blind append without duplicate check.
    // Duplicates are resolved (last-write-wins) during dms_sort_coo().
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
      dms_invalidate_csc_cache(matrix);
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
  dms_invalidate_csc_cache(mat);

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
  dms_invalidate_csc_cache(mat);
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
