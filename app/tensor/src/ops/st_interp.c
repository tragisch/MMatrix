/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

/* Graph-interpreter tensor kernels: broadcast binary operations.
 *
 * The reference semantics are the scalar loops of nm_lib's graph
 * interpreter (run_binary in nm_graph_interp.c): NumPy broadcasting over
 * any axes, fast paths for equal shapes, scalars, and shape suffixes, and
 * a strided walk with vectorized row kernels for everything else.
 */

#include "st_interp.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
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
#include <Accelerate/Accelerate.h>
#endif

#include "st_bf16_utils.h"
#include "st_gpu_guard.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define ST_INTERP_HAVE_NEON 1
#endif

/* OpenMP is only used above this many output elements. bench_st_binary's
 * thread sweep on this 4-performance-core machine shows the fork/join cost
 * regressing small and mid inputs (0.15x-0.9x below ~1M elements) and only
 * marginal gains above ~2M, so the threshold keeps realistic transformer
 * tensor sizes on the sequential path (same rationale as sv_optim). */
#define ST_BIN_OMP_THRESHOLD 2097152u

/* ---- Walk machinery (adapted from nm_lib's interpreter) ---------------- *
 *
 * Walks a row-major index space of rank ST_WALK_DIMS and keeps up to three
 * operand offsets in step, each with its own strides. walk_finish drops
 * dimensions of size one and merges dimensions every operand reads
 * contiguously, so the kernels loop over rows of inner elements. */

enum { ST_WALK_DIMS = ST_MAX_DIMS };
enum { ST_WALK_OPERANDS = 3 };

typedef struct {
  size_t d[ST_WALK_DIMS];
  size_t c[ST_WALK_DIMS];
  ptrdiff_t stride[ST_WALK_OPERANDS][ST_WALK_DIMS];
  ptrdiff_t offset[ST_WALK_OPERANDS];
  size_t operands;
  size_t rank;
} StWalk;

/* Left-pads `ndim/shape` with ones to rank ST_WALK_DIMS. */
static void st_walk_pad(size_t ndim, const size_t *shape, size_t out[ST_WALK_DIMS]) {
  const size_t offset = ST_WALK_DIMS - ndim;
  for (size_t i = 0; i < ST_WALK_DIMS; ++i) {
    out[i] = i < offset ? 1 : shape[i - offset];
  }
}

/* Element strides of `ndim/shape` padded to rank ST_WALK_DIMS, zero along
 * the dimensions that broadcast to `d`. */
static void st_walk_broadcast_strides(size_t ndim, const size_t *shape,
                                      const size_t d[ST_WALK_DIMS],
                                      ptrdiff_t out[ST_WALK_DIMS]) {
  size_t padded[ST_WALK_DIMS];
  st_walk_pad(ndim, shape, padded);
  ptrdiff_t running = 1;
  for (size_t i = ST_WALK_DIMS; i-- > 0;) {
    out[i] = padded[i] == 1 && d[i] != 1 ? 0 : running;
    running *= (ptrdiff_t)padded[i];
  }
}

static void st_walk_init(StWalk *w, size_t ndim, const size_t *shape,
                         size_t operands) {
  memset(w, 0, sizeof(*w));
  st_walk_pad(ndim, shape, w->d);
  w->operands = operands;
}

static void st_walk_finish(StWalk *w) {
  size_t rank = 0;
  for (size_t i = 0; i < ST_WALK_DIMS; ++i) {
    if (w->d[i] == 1) {
      continue;
    }
    bool merge = rank > 0;
    for (size_t j = 0; merge && j < w->operands; ++j) {
      merge = w->stride[j][rank - 1] == w->stride[j][i] * (ptrdiff_t)w->d[i];
    }
    if (merge) {
      w->d[rank - 1] *= w->d[i];
      for (size_t j = 0; j < w->operands; ++j) {
        w->stride[j][rank - 1] = w->stride[j][i];
      }
      continue;
    }
    w->d[rank] = w->d[i];
    for (size_t j = 0; j < w->operands; ++j) {
      w->stride[j][rank] = w->stride[j][i];
    }
    rank++;
  }
  if (rank == 0) {
    w->d[0] = 1;
    for (size_t j = 0; j < w->operands; ++j) {
      w->stride[j][0] = 0;
    }
    rank = 1;
  }
  w->rank = rank;
}

static size_t st_walk_inner(const StWalk *w) { return w->d[w->rank - 1]; }

static size_t st_walk_rows(const StWalk *w) {
  size_t rows = 1;
  for (size_t i = 0; i + 1 < w->rank; ++i) {
    rows *= w->d[i];
  }
  return rows;
}

/* Operand offsets at the start of walk row `row` (fixed radix order, the
 * last axis is the fastest). Used once per parallel block; within a block
 * the walk advances incrementally. */
static void st_walk_row_offsets(const StWalk *w, size_t row,
                                ptrdiff_t *offsets) {
  size_t rest = row;
  for (size_t j = 0; j < w->operands; ++j) {
    offsets[j] = 0;
  }
  for (size_t k = w->rank - 1; k-- > 0;) {
    const size_t digit = rest % w->d[k];
    rest /= w->d[k];
    for (size_t j = 0; j < w->operands; ++j) {
      offsets[j] += (ptrdiff_t)digit * w->stride[j][k];
    }
  }
}

/* Row coordinates of `row`, so walk_next_row can advance from it. */
static void st_walk_row_digits(const StWalk *w, size_t row, size_t *digits) {
  size_t rest = row;
  for (size_t k = 0; k < ST_WALK_DIMS; ++k) {
    digits[k] = 0;
  }
  for (size_t k = w->rank - 1; k-- > 0;) {
    digits[k] = rest % w->d[k];
    rest /= w->d[k];
  }
}

/* Advance to the next walk row, odometer style (last axis fastest). */
static void st_walk_next_row(StWalk *w) {
  for (size_t k = w->rank - 1; k-- > 0;) {
    for (size_t j = 0; j < w->operands; ++j) {
      w->offset[j] += w->stride[j][k];
    }
    if (++w->c[k] < w->d[k]) {
      return;
    }
    for (size_t j = 0; j < w->operands; ++j) {
      w->offset[j] -= w->stride[j][k] * (ptrdiff_t)w->d[k];
    }
    w->c[k] = 0;
  }
}

/* ---- Shared input checks ---------------------------------------------- */

#ifndef ST_LOG_ERROR
#define ST_LOG_ERROR(fmt, ...) \
  fprintf(stderr, "[st][error] " fmt "\n", ##__VA_ARGS__)
#endif

static bool st_interp_tensor_valid(const FloatTensor *t) {
  if (t == NULL || t->values == NULL || t->numel == 0 || t->ndim == 0 ||
      t->ndim > ST_MAX_DIMS || !st_is_contiguous(t)) {
    return false;
  }
  size_t expected_numel = 0;
  return st_numel_from_shape(t->ndim, t->shape, &expected_numel) &&
         expected_numel == t->numel;
}

/* ---- Broadcast helpers ------------------------------------------------- */

/* NumPy-style broadcast of two shapes; right-aligned, size 1 expands. */
static bool st_interp_broadcast_shape(size_t a_ndim, const size_t *a_dims,
                                      size_t b_ndim, const size_t *b_dims,
                                      size_t *out_ndim, size_t *out_dims) {
  const size_t ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
  for (size_t i = 0; i < ndim; ++i) {
    const size_t da = i < ndim - a_ndim ? 1 : a_dims[i - (ndim - a_ndim)];
    const size_t db = i < ndim - b_ndim ? 1 : b_dims[i - (ndim - b_ndim)];
    if (da != db && da != 1 && db != 1) {
      return false;
    }
    out_dims[i] = da == 1 ? db : da;
  }
  *out_ndim = ndim;
  return true;
}

static size_t st_interp_shape_numel(size_t ndim, const size_t *dims) {
  size_t numel = 1;
  for (size_t i = 0; i < ndim; ++i) {
    numel *= dims[i];
  }
  return numel;
}

/* ---- Binary row kernels ------------------------------------------------- *
 *
 * The operation is hoisted out of the loop so the inner loop vectorizes.
 * `sa`/`sb` select contiguous (a[i]) or broadcast (a[0]) operand access.
 * OpenMP is applied by the callers over rows/blocks, never nested. */

static void st_bin_row(StBinOp op, const float *a, const float *b, float *out,
                       size_t n) {
#if defined(USE_ACCELERATE)
  /* Division stays on the C path: vDSP_vdiv is not bit-identical to IEEE
   * scalar division (1 ulp differences), and the binary kernels promise
   * bit-exact results. Add/sub/mul are single IEEE ops and exact in vDSP. */
  if (n >= 32 && op != ST_BIN_DIV) {
    switch (op) {
      case ST_BIN_ADD:
        vDSP_vadd(a, 1, b, 1, out, 1, (vDSP_Length)n);
        return;
      case ST_BIN_SUB:
        vDSP_vsub(a, 1, b, 1, out, 1, (vDSP_Length)n);
        return;
      case ST_BIN_MUL:
        vDSP_vmul(a, 1, b, 1, out, 1, (vDSP_Length)n);
        return;
      default:
        break;
    }
  }
#endif
  switch (op) {
    case ST_BIN_ADD:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
      }
      return;
    case ST_BIN_SUB:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
      }
      return;
    case ST_BIN_MUL:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
      }
      return;
    case ST_BIN_DIV:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
      }
      return;
  }
}

/* Division by a loop-invariant scalar, IEEE-exact per element.
 *
 * -ffast-math lets the compiler rewrite `a[i] / c` into `a[i] * (1/c)` with a
 * once-rounded reciprocal (double rounding, 1 ulp off), even through NEON
 * splat intrinsics. clang: `#pragma float_control(precise, on)` disables the
 * transform for this function while keeping -O3; gcc: the no-fast-math
 * optimize attribute, same pattern as the NaN check in sm_kernels.c. */
#if defined(__clang__)
#pragma float_control(push)
#pragma float_control(precise, on)
#elif defined(__GNUC__)
__attribute__((optimize("no-fast-math")))
#endif
static void st_div_scalar_b_exact(const float *a, const float *b, float *out,
                                  size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = a[i] / b[0];
  }
}
#if defined(__clang__)
#pragma float_control(pop)
#endif

static void st_bin_row_scalar_b(StBinOp op, const float *a, const float *b,
                                float *out, size_t n) {
  switch (op) {
    case ST_BIN_ADD:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[0];
      }
      return;
    case ST_BIN_SUB:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[0];
      }
      return;
    case ST_BIN_MUL:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[0];
      }
      return;
    case ST_BIN_DIV:
      /* Exact division: see st_div_scalar_b_exact. */
      st_div_scalar_b_exact(a, b, out, n);
      return;
  }
}

static void st_bin_row_scalar_a(StBinOp op, const float *a, const float *b,
                                float *out, size_t n) {
  switch (op) {
    case ST_BIN_ADD:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[0] + b[i];
      }
      return;
    case ST_BIN_SUB:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[0] - b[i];
      }
      return;
    case ST_BIN_MUL:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[0] * b[i];
      }
      return;
    case ST_BIN_DIV:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[0] / b[i];
      }
      return;
  }
}

static void st_bin_row_strided(StBinOp op, const float *a, ptrdiff_t ta,
                               const float *b, ptrdiff_t tb, float *out,
                               size_t n) {
  switch (op) {
    case ST_BIN_ADD:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[(ptrdiff_t)i * ta] + b[(ptrdiff_t)i * tb];
      }
      return;
    case ST_BIN_SUB:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[(ptrdiff_t)i * ta] - b[(ptrdiff_t)i * tb];
      }
      return;
    case ST_BIN_MUL:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[(ptrdiff_t)i * ta] * b[(ptrdiff_t)i * tb];
      }
      return;
    case ST_BIN_DIV:
#pragma omp simd
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[(ptrdiff_t)i * ta] / b[(ptrdiff_t)i * tb];
      }
      return;
  }
}

/* ---- Binary core on raw f32 buffers ------------------------------------ */

static bool st_binary_f32(StBinOp op, const float *a, size_t a_ndim,
                          const size_t *a_dims, const float *b, size_t b_ndim,
                          const size_t *b_dims, float *out, size_t o_ndim,
                          const size_t *o_dims) {
  const size_t n = st_interp_shape_numel(o_ndim, o_dims);
  const size_t na = st_interp_shape_numel(a_ndim, a_dims);
  const size_t nb = st_interp_shape_numel(b_ndim, b_dims);

  const bool use_omp = n > ST_BIN_OMP_THRESHOLD && omp_get_max_threads() > 1;

  if (na == n && nb == n) {
    if (use_omp) {
#pragma omp parallel
      {
        const size_t threads = (size_t)omp_get_num_threads();
        const size_t thread = (size_t)omp_get_thread_num();
        const size_t first = n * thread / threads;
        const size_t last = n * (thread + 1) / threads;
        if (last > first) {
          st_bin_row(op, a + first, b + first, out + first, last - first);
        }
      }
    } else {
      st_bin_row(op, a, b, out, n);
    }
    return true;
  }
  if (na == n && nb == 1) {
    if (use_omp) {
#pragma omp parallel
      {
        const size_t threads = (size_t)omp_get_num_threads();
        const size_t thread = (size_t)omp_get_thread_num();
        const size_t first = n * thread / threads;
        const size_t last = n * (thread + 1) / threads;
        if (last > first) {
          st_bin_row_scalar_b(op, a + first, b, out + first, last - first);
        }
      }
    } else {
      st_bin_row_scalar_b(op, a, b, out, n);
    }
    return true;
  }
  if (na == n && nb > 0 && n % nb == 0 && b_ndim <= o_ndim) {
    /* b broadcasts along leading dimensions only (bias, scale): b's shape
     * is a suffix of the output shape. */
    bool suffix = true;
    for (size_t i = 0; i < b_ndim; ++i) {
      suffix = suffix && b_dims[i] == o_dims[o_ndim - b_ndim + i];
    }
    if (suffix) {
      const size_t rows = n / nb;
      if (use_omp) {
#pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r) {
          st_bin_row(op, a + r * nb, b, out + r * nb, nb);
        }
      } else {
        for (size_t r = 0; r < rows; ++r) {
          st_bin_row(op, a + r * nb, b, out + r * nb, nb);
        }
      }
      return true;
    }
  }

  /* General case: strided walk with row kernels. */
  StWalk w;
  st_walk_init(&w, o_ndim, o_dims, 2);
  st_walk_broadcast_strides(a_ndim, a_dims, w.d, w.stride[0]);
  st_walk_broadcast_strides(b_ndim, b_dims, w.d, w.stride[1]);
  st_walk_finish(&w);
  const size_t inner = st_walk_inner(&w);
  const size_t rows = st_walk_rows(&w);
  const ptrdiff_t ta = w.stride[0][w.rank - 1];
  const ptrdiff_t tb = w.stride[1][w.rank - 1];
  if (use_omp && rows > 1) {
    /* Parallel blocks; each block walks incrementally from its first row. */
    const size_t row_block = rows / 8 + 1;
#pragma omp parallel for schedule(static)
    for (size_t block = 0; block < rows; block += row_block) {
      StWalk local = w;
      st_walk_row_digits(&w, block, local.c);
      st_walk_row_offsets(&w, block, local.offset);
      const size_t block_end =
          block + row_block < rows ? block + row_block : rows;
      float *po = out + block * inner;
      for (size_t r = block; r < block_end;
           ++r, st_walk_next_row(&local), po += inner) {
        const float *pa = a + local.offset[0];
        const float *pb = b + local.offset[1];
        if (ta == 1 && tb == 1) {
          st_bin_row(op, pa, pb, po, inner);
        } else if (ta == 1 && tb == 0) {
          st_bin_row_scalar_b(op, pa, pb, po, inner);
        } else if (ta == 0 && tb == 1) {
          st_bin_row_scalar_a(op, pa, pb, po, inner);
        } else {
          st_bin_row_strided(op, pa, ta, pb, tb, po, inner);
        }
      }
    }
    return true;
  }
  for (size_t r = 0; r < rows; ++r, st_walk_next_row(&w), out += inner) {
    const float *pa = a + w.offset[0];
    const float *pb = b + w.offset[1];
    if (ta == 1 && tb == 1) {
      st_bin_row(op, pa, pb, out, inner);
    } else if (ta == 1 && tb == 0) {
      st_bin_row_scalar_b(op, pa, pb, out, inner);
    } else if (ta == 0 && tb == 1) {
      st_bin_row_scalar_a(op, pa, pb, out, inner);
    } else {
      st_bin_row_strided(op, pa, ta, pb, tb, out, inner);
    }
  }
  return true;
}

/* ---- Public API -------------------------------------------------------- */

bool st_binary_broadcast_into(StBinOp op, const FloatTensor *a,
                              const FloatTensor *b, FloatTensor *out) {
  if (op != ST_BIN_ADD && op != ST_BIN_SUB && op != ST_BIN_MUL &&
      op != ST_BIN_DIV) {
    ST_LOG_ERROR("st_binary_broadcast_into: unknown op %d.", (int)op);
    return false;
  }
  if (!st_interp_tensor_valid(a) || !st_interp_tensor_valid(b) ||
      !st_interp_tensor_valid(out)) {
    ST_LOG_ERROR("st_binary_broadcast_into: invalid tensor input.");
    return false;
  }
  ST_ASSERT_NOT_PENDING(a);
  ST_ASSERT_NOT_PENDING(b);
  ST_ASSERT_NOT_PENDING(out);

  size_t broadcast_ndim = 0;
  size_t broadcast_dims[ST_MAX_DIMS];
  if (!st_interp_broadcast_shape(a->ndim, a->shape, b->ndim, b->shape,
                                 &broadcast_ndim, broadcast_dims)) {
    ST_LOG_ERROR("st_binary_broadcast_into: shapes do not broadcast.");
    return false;
  }
  if (broadcast_ndim != out->ndim ||
      memcmp(broadcast_dims, out->shape, out->ndim * sizeof(size_t)) != 0) {
    ST_LOG_ERROR("st_binary_broadcast_into: output shape mismatch.");
    return false;
  }

  /* bf16 operands promote to f32; a bf16 output is rounded back at the end. */
  FloatTensor *a_f32 = a->dtype == ST_DTYPE_BF16 ? st_to_f32(a) : NULL;
  FloatTensor *b_f32 = b->dtype == ST_DTYPE_BF16 ? st_to_f32(b) : NULL;
  if ((a->dtype == ST_DTYPE_BF16 && a_f32 == NULL) ||
      (b->dtype == ST_DTYPE_BF16 && b_f32 == NULL)) {
    st_destroy(a_f32);
    st_destroy(b_f32);
    return false;
  }
  const float *a_values = a_f32 ? a_f32->values : a->values;
  const float *b_values = b_f32 ? b_f32->values : b->values;

  FloatTensor *out_f32 = NULL;
  float *out_values = out->values;
  if (out->dtype == ST_DTYPE_BF16) {
    out_f32 = st_create(out->ndim, out->shape);
    if (out_f32 == NULL) {
      st_destroy(a_f32);
      st_destroy(b_f32);
      return false;
    }
    out_values = out_f32->values;
  }

  const bool ok = st_binary_f32(op, a_values, a->ndim, a->shape, b_values,
                                b->ndim, b->shape, out_values, out->ndim,
                                out->shape);
  if (ok && out_f32 != NULL) {
    st_f32_to_bf16_bulk(out_f32->values, (uint16_t *)out->values,
                        out->numel);
  }
  st_destroy(out_f32);
  st_destroy(a_f32);
  st_destroy(b_f32);
  return ok;
}

FloatTensor *st_binary_broadcast(StBinOp op, const FloatTensor *a,
                                 const FloatTensor *b) {
  if (op != ST_BIN_ADD && op != ST_BIN_SUB && op != ST_BIN_MUL &&
      op != ST_BIN_DIV) {
    return NULL;
  }
  if (!st_interp_tensor_valid(a) || !st_interp_tensor_valid(b)) {
    ST_LOG_ERROR("st_binary_broadcast: invalid tensor input.");
    return NULL;
  }
  ST_ASSERT_NOT_PENDING(a);
  ST_ASSERT_NOT_PENDING(b);

  size_t out_ndim = 0;
  size_t out_dims[ST_MAX_DIMS];
  if (!st_interp_broadcast_shape(a->ndim, a->shape, b->ndim, b->shape,
                                 &out_ndim, out_dims)) {
    ST_LOG_ERROR("st_binary_broadcast: shapes do not broadcast.");
    return NULL;
  }
  FloatTensor *out = st_create(out_ndim, out_dims);
  if (out == NULL) {
    return NULL;
  }
  if (!st_binary_broadcast_into(op, a, b, out)) {
    st_destroy(out);
    return NULL;
  }
  return out;
}
