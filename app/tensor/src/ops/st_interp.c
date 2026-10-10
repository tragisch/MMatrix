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

#include "sm.h"
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

/* ---- LayerNorm ---------------------------------------------------------- */

/* OpenMP over groups only above this many input elements; see
 * bench_st_norm_reduce for the measured threshold. */
#define ST_LAYERNORM_OMP_THRESHOLD 262144u

/* Bit-level finiteness check (-ffast-math breaks isnan/isfinite), same
 * pattern as sv_optim.c. */
static bool st_interp_float_finite(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7f800000u) != 0x7f800000u;
}

/* Sum of `n` floats with eight partial sums, so the loop vectorizes and the
 * result is independent of the OpenMP thread count (nm_lib's sum8). */
static float st_interp_sum8(const float *x, size_t n) {
  float lanes[8] = {0.0f};
  size_t j = 0;
  for (; j + 8 <= n; j += 8) {
    for (size_t l = 0; l < 8; ++l) {
      lanes[l] += x[j + l];
    }
  }
  float sum = 0.0f;
  for (; j < n; ++j) {
    sum += x[j];
  }
  for (size_t l = 0; l < 8; ++l) {
    sum += lanes[l];
  }
  return sum;
}

/* One normalization group, verbatim run_layernorm from nm_lib: mean via
 * sum8, centered values into y, variance via eight partial sums, then
 * scale and bias. Safe in-place when in == out. */
static void st_layernorm_group(const float *in, const float *scale,
                               const float *bias, size_t group, float eps,
                               float *out) {
  const float mean = st_interp_sum8(in, group) / (float)group;
  for (size_t j = 0; j < group; ++j) {
    out[j] = in[j] - mean;
  }
  float lanes[8] = {0.0f};
  size_t j = 0;
  for (; j + 8 <= group; j += 8) {
    for (size_t l = 0; l < 8; ++l) {
      lanes[l] += out[j + l] * out[j + l];
    }
  }
  float variance = 0.0f;
  for (; j < group; ++j) {
    variance += out[j] * out[j];
  }
  for (size_t l = 0; l < 8; ++l) {
    variance += lanes[l];
  }
  variance /= (float)group;
  const float inverse = 1.0f / sqrtf(variance + eps);
  if (scale != NULL && bias != NULL) {
#pragma omp simd
    for (size_t i = 0; i < group; ++i) {
      out[i] = out[i] * inverse * scale[i] + bias[i];
    }
  } else {
    for (size_t i = 0; i < group; ++i) {
      float value = out[i] * inverse;
      if (scale != NULL) {
        value *= scale[i];
      }
      if (bias != NULL) {
        value += bias[i];
      }
      out[i] = value;
    }
  }
}

static bool st_layernorm_f32(const float *in, const float *scale,
                             const float *bias, size_t group, size_t groups,
                             float eps, float *out) {
  const size_t total = groups * group;
  if (total > ST_LAYERNORM_OMP_THRESHOLD && omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
    for (size_t g = 0; g < groups; ++g) {
      st_layernorm_group(in + g * group, scale, bias, group, eps,
                         out + g * group);
    }
  } else {
    for (size_t g = 0; g < groups; ++g) {
      st_layernorm_group(in + g * group, scale, bias, group, eps,
                         out + g * group);
    }
  }
  return true;
}

bool st_layernorm(const FloatTensor *x, const FloatTensor *gamma,
                  const FloatTensor *beta, size_t norm_rank, float eps,
                  FloatTensor *out) {
  if (!st_interp_tensor_valid(x) || !st_interp_tensor_valid(out)) {
    ST_LOG_ERROR("st_layernorm: invalid tensor input.");
    return false;
  }
  ST_ASSERT_NOT_PENDING(x);
  ST_ASSERT_NOT_PENDING(out);
  if (out->dtype != ST_DTYPE_F32 && out->dtype != ST_DTYPE_BF16) {
    return false;
  }
  if (out->ndim != x->ndim ||
      memcmp(out->shape, x->shape, x->ndim * sizeof(size_t)) != 0) {
    ST_LOG_ERROR("st_layernorm: output shape mismatch.");
    return false;
  }
  if (norm_rank == 0 || norm_rank > x->ndim) {
    ST_LOG_ERROR("st_layernorm: norm_rank out of range.");
    return false;
  }
  if (!st_interp_float_finite(eps) || eps < 0.0f) {
    ST_LOG_ERROR("st_layernorm: eps must be finite and non-negative.");
    return false;
  }

  size_t group = 1;
  for (size_t i = x->ndim - norm_rank; i < x->ndim; ++i) {
    group *= x->shape[i];
  }
  const size_t groups = x->numel / group;

  /* gamma/beta are optional; when present they must cover one group. */
  const FloatTensor *scale = gamma;
  const FloatTensor *shift = beta;
  if (scale != NULL && !st_interp_tensor_valid(scale)) {
    ST_LOG_ERROR("st_layernorm: invalid gamma tensor.");
    return false;
  }
  if (shift != NULL && !st_interp_tensor_valid(shift)) {
    ST_LOG_ERROR("st_layernorm: invalid beta tensor.");
    return false;
  }
  if (scale != NULL && scale->numel != group) {
    ST_LOG_ERROR("st_layernorm: gamma must have one element per group.");
    return false;
  }
  if (shift != NULL && shift->numel != group) {
    ST_LOG_ERROR("st_layernorm: beta must have one element per group.");
    return false;
  }
  if (scale != NULL) {
    ST_ASSERT_NOT_PENDING(scale);
  }
  if (shift != NULL) {
    ST_ASSERT_NOT_PENDING(shift);
  }

  /* bf16 promotes to f32; a bf16 output is rounded back at the end. */
  FloatTensor *x_f32 = x->dtype == ST_DTYPE_BF16 ? st_to_f32(x) : NULL;
  FloatTensor *gamma_f32 =
      scale != NULL && scale->dtype == ST_DTYPE_BF16 ? st_to_f32(scale) : NULL;
  FloatTensor *beta_f32 =
      shift != NULL && shift->dtype == ST_DTYPE_BF16 ? st_to_f32(shift) : NULL;
  if ((x->dtype == ST_DTYPE_BF16 && x_f32 == NULL) ||
      (scale != NULL && scale->dtype == ST_DTYPE_BF16 && gamma_f32 == NULL) ||
      (shift != NULL && shift->dtype == ST_DTYPE_BF16 && beta_f32 == NULL)) {
    st_destroy(beta_f32);
    st_destroy(gamma_f32);
    st_destroy(x_f32);
    return false;
  }
  const float *in_values = x_f32 ? x_f32->values : x->values;
  const float *scale_values =
      scale == NULL ? NULL : (gamma_f32 ? gamma_f32->values : scale->values);
  const float *shift_values =
      shift == NULL ? NULL : (beta_f32 ? beta_f32->values : shift->values);

  FloatTensor *out_f32 = NULL;
  float *out_values = out->values;
  if (out->dtype == ST_DTYPE_BF16) {
    out_f32 = st_create(out->ndim, out->shape);
    if (out_f32 == NULL) {
      st_destroy(beta_f32);
      st_destroy(gamma_f32);
      st_destroy(x_f32);
      return false;
    }
    out_values = out_f32->values;
  }

  const bool ok = st_layernorm_f32(in_values, scale_values, shift_values,
                                   group, groups, eps, out_values);
  if (ok && out_f32 != NULL) {
    st_f32_to_bf16_bulk(out_f32->values, (uint16_t *)out->values,
                        out->numel);
  }
  st_destroy(out_f32);
  st_destroy(beta_f32);
  st_destroy(gamma_f32);
  st_destroy(x_f32);
  return ok;
}

/* ---- Reduce max ---------------------------------------------------------- */

/* Parallel over output slots only above this many input elements.
 * bench_st_norm_reduce measured no gain at 131k elements on this
 * 4-performance-core machine (fork/join cost of a one-op-per-element fold),
 * so realistic sizes stay sequential, like the Phase 1 optimizer kernels. */
#define ST_REDUCE_OMP_THRESHOLD 1048576u

/* Horizontal max over the eight fold lanes. noinline forces the vectorizer
 * to load all eight lanes from memory: when some lanes stay in registers,
 * clang pads the horizontal fmaxnm with -FLT_MAX (under -ffinite-math-only
 * it assumes nothing can be smaller), which would beat real -INFINITY
 * inputs. */
__attribute__((noinline)) static float st_max8_lanes(const float *lanes) {
  float best = lanes[0];
  for (size_t l = 1; l < 8; ++l) {
    best = best > lanes[l] ? best : lanes[l];
  }
  return best;
}

/* Fold one contiguous run: eight independent lanes track the maximum (the
 * OpenMP max reduction is not usable here because its private lanes start
 * at -FLT_MAX, which would swallow all -INFINITY inputs), and an integer
 * side channel records whether NaN bits appeared; a NaN rescan then
 * returns the first NaN in scan order. Bit checks on the integer pattern
 * stay meaningful under -ffast-math (no float comparisons on NaN). */
static float st_reduce_max_run(const float *x, size_t n) {
  float lanes[8];
  for (size_t l = 0; l < 8; ++l) {
    lanes[l] = -INFINITY;
  }
  uint32_t any_nan = 0;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    for (size_t l = 0; l < 8; ++l) {
      uint32_t bits = 0;
      memcpy(&bits, &x[i + l], sizeof(bits));
      any_nan |= (((bits & 0x7f800000u) == 0x7f800000u) &
                  ((bits & 0x007fffffu) != 0u))
                     ? 1u
                     : 0u;
      lanes[l] = lanes[l] > x[i + l] ? lanes[l] : x[i + l];
    }
  }
  for (; i < n; ++i) {
    uint32_t bits = 0;
    memcpy(&bits, &x[i], sizeof(bits));
    any_nan |= (((bits & 0x7f800000u) == 0x7f800000u) &
                ((bits & 0x007fffffu) != 0u))
                   ? 1u
                   : 0u;
    lanes[0] = lanes[0] > x[i] ? lanes[0] : x[i];
  }
  const float best = st_max8_lanes(lanes);
  if (any_nan != 0u) {
    for (size_t j = 0; j < n; ++j) {
      uint32_t bits = 0;
      memcpy(&bits, &x[j], sizeof(bits));
      if (((bits & 0x7f800000u) == 0x7f800000u) &&
          ((bits & 0x007fffffu) != 0u)) {
        return x[j];
      }
    }
  }
  return best;
}

/* Vectorizable NaN detection over a contiguous buffer: the fold kernels
 * then only need the careful path when a NaN is actually present. */
static bool st_interp_any_nan(const float *in, size_t n) {
  uint32_t any_nan = 0;
#pragma omp simd reduction(| : any_nan)
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits = 0;
    memcpy(&bits, &in[i], sizeof(bits));
    any_nan |= (((bits & 0x7f800000u) == 0x7f800000u) &
                ((bits & 0x007fffffu) != 0u))
                   ? 1u
                   : 0u;
  }
  return any_nan != 0u;
}

/* One careful walk row for the NaN path: the first NaN per slot in scan
 * order sticks, later elements leave it in place. */
static void st_reduce_max_walk_row_careful(const float *x, float *y,
                                           ptrdiff_t to, size_t inner) {
  if (to == 0) {
    for (size_t i = 0; i < inner; ++i) {
      uint32_t bits = 0;
      memcpy(&bits, &x[i], sizeof(bits));
      if (((bits & 0x7f800000u) == 0x7f800000u) &&
          ((bits & 0x007fffffu) != 0u)) {
        *y = x[i];
        return;
      }
      if (x[i] > *y) {
        *y = x[i];
      }
    }
    return;
  }
  for (size_t i = 0; i < inner; ++i) {
    float *slot = y + (ptrdiff_t)i * to;
    uint32_t slot_bits = 0;
    memcpy(&slot_bits, slot, sizeof(slot_bits));
    if (((slot_bits & 0x7f800000u) == 0x7f800000u) &&
        ((slot_bits & 0x007fffffu) != 0u)) {
      continue;  /* the first NaN already won */
    }
    uint32_t bits = 0;
    memcpy(&bits, &x[i], sizeof(bits));
    if (((bits & 0x7f800000u) == 0x7f800000u) &&
        ((bits & 0x007fffffu) != 0u)) {
      *slot = x[i];
    } else if (x[i] > *slot) {
      *slot = x[i];
    }
  }
}

/* Sequential nm_lib-style walk for masks that reduce non-trailing axes: the
 * input is visited in memory order and folds into output slots that stand
 * still along reduced axes. Without NaNs this is the plain reference fold;
 * a single vectorized pre-scan decides, and the careful walk runs only
 * when a NaN is present. */
static void st_reduce_max_walk(const float *in, size_t ndim,
                               const size_t *dims, uint32_t mask,
                               size_t in_numel, float *out,
                               size_t out_numel) {
  size_t kept_dims[ST_MAX_DIMS];
  for (size_t a = 0; a < ndim; ++a) {
    kept_dims[a] = ((mask >> a) & 1u) != 0u ? 1 : dims[a];
  }
  StWalk w;
  st_walk_init(&w, ndim, dims, 2);
  st_walk_broadcast_strides(ndim, kept_dims, w.d, w.stride[0]);
  st_walk_broadcast_strides(ndim, dims, w.d, w.stride[1]);
  st_walk_finish(&w);
  const size_t inner = st_walk_inner(&w);
  const size_t rows = st_walk_rows(&w);
  const ptrdiff_t to = w.stride[0][w.rank - 1];

  if (st_interp_any_nan(in, in_numel)) {
    for (size_t i = 0; i < out_numel; ++i) {
      out[i] = -INFINITY;
    }
    for (size_t r = 0; r < rows; ++r, st_walk_next_row(&w)) {
      st_reduce_max_walk_row_careful(in + w.offset[1], out + w.offset[0], to,
                                     inner);
    }
    return;
  }

  for (size_t i = 0; i < out_numel; ++i) {
    out[i] = -INFINITY;
  }
  for (size_t r = 0; r < rows; ++r, st_walk_next_row(&w)) {
    const float *x = in + w.offset[1];
    float *y = out + w.offset[0];
    if (to == 0) {
      float max = *y;
      for (size_t i = 0; i < inner; ++i) {
        max = x[i] > max ? x[i] : max;
      }
      *y = max;
    } else {
      for (size_t i = 0; i < inner; ++i) {
        float *slot = y + (ptrdiff_t)i * to;
        *slot = x[i] > *slot ? x[i] : *slot;
      }
    }
  }
}

static void st_reduce_max_f32(const float *in, size_t ndim,
                              const size_t *dims, uint32_t mask, float *out,
                              size_t out_numel) {
  if (mask == 0u) {
    memcpy(out, in, out_numel * sizeof(float));
    return;
  }

  size_t reduced = 0;
  size_t in_numel = 1;
  for (size_t a = 0; a < ndim; ++a) {
    reduced += (mask >> a) & 1u;
    in_numel *= dims[a];
  }
  const uint32_t suffix_mask =
      (uint32_t)(((1u << reduced) - 1u) << (ndim - reduced));
  if (mask != suffix_mask) {
    /* Non-trailing reduced axes: strided access pattern, sequential walk. */
    st_reduce_max_walk(in, ndim, dims, mask, in_numel, out, out_numel);
    return;
  }

  /* Trailing reduced axes: every output slot is one contiguous run, and the
   * slot base is simply the slot index times the run length. */
  size_t chunk = 1;
  for (size_t a = ndim - reduced; a < ndim; ++a) {
    chunk *= dims[a];
  }
  const size_t slots = out_numel;
  const bool use_omp = slots > 1 && slots * chunk > ST_REDUCE_OMP_THRESHOLD &&
                       omp_get_max_threads() > 1;
  if (use_omp) {
#pragma omp parallel for schedule(static)
    for (size_t slot = 0; slot < slots; ++slot) {
      out[slot] = st_reduce_max_run(in + slot * chunk, chunk);
    }
    return;
  }
  for (size_t slot = 0; slot < slots; ++slot) {
    out[slot] = st_reduce_max_run(in + slot * chunk, chunk);
  }
}

bool st_reduce_max_into(const FloatTensor *x, uint32_t axes_mask,
                        bool keepdims, FloatTensor *out) {
  if (!st_interp_tensor_valid(x) || !st_interp_tensor_valid(out)) {
    ST_LOG_ERROR("st_reduce_max_into: invalid tensor input.");
    return false;
  }
  ST_ASSERT_NOT_PENDING(x);
  ST_ASSERT_NOT_PENDING(out);
  if (x->ndim > 32 || (axes_mask >> x->ndim) != 0u) {
    ST_LOG_ERROR("st_reduce_max_into: axes_mask out of range.");
    return false;
  }

  size_t expected[ST_MAX_DIMS];
  size_t expected_ndim = 0;
  const uint32_t reduced =
      axes_mask & ((x->ndim >= 32) ? 0xffffffffu : ((1u << x->ndim) - 1u));
  for (size_t a = 0; a < x->ndim; ++a) {
    const bool is_reduced = ((reduced >> a) & 1u) != 0u;
    if (is_reduced && keepdims) {
      expected[expected_ndim++] = 1;
    } else if (!is_reduced) {
      expected[expected_ndim++] = x->shape[a];
    }
  }
  const bool shape_ok =
      (expected_ndim == out->ndim &&
       memcmp(expected, out->shape, expected_ndim * sizeof(size_t)) == 0) ||
      /* Reducing every axis yields a scalar; tensors keep it as [1]. */
      (expected_ndim == 0 && out->ndim == 1 && out->shape[0] == 1);
  if (!shape_ok) {
    ST_LOG_ERROR("st_reduce_max_into: output shape mismatch.");
    return false;
  }

  /* bf16 promotes to f32; a bf16 output is rounded back at the end. */
  FloatTensor *x_f32 = x->dtype == ST_DTYPE_BF16 ? st_to_f32(x) : NULL;
  if (x->dtype == ST_DTYPE_BF16 && x_f32 == NULL) {
    return false;
  }
  const float *in_values = x_f32 ? x_f32->values : x->values;

  FloatTensor *out_f32 = NULL;
  float *out_values = out->values;
  if (out->dtype == ST_DTYPE_BF16) {
    out_f32 = st_create(out->ndim, out->shape);
    if (out_f32 == NULL) {
      st_destroy(x_f32);
      return false;
    }
    out_values = out_f32->values;
  }

  st_reduce_max_f32(in_values, x->ndim, x->shape, axes_mask, out_values,
                    out->numel);
  if (out_f32 != NULL) {
    st_f32_to_bf16_bulk(out_f32->values, (uint16_t *)out->values,
                        out->numel);
  }
  st_destroy(out_f32);
  st_destroy(x_f32);
  return true;
}

FloatTensor *st_reduce_max(const FloatTensor *x, uint32_t axes_mask,
                           bool keepdims) {
  if (!st_interp_tensor_valid(x)) {
    ST_LOG_ERROR("st_reduce_max: invalid tensor input.");
    return NULL;
  }
  ST_ASSERT_NOT_PENDING(x);
  if (x->ndim > 32 || (axes_mask >> x->ndim) != 0u) {
    return NULL;
  }

  size_t out_dims[ST_MAX_DIMS];
  size_t out_ndim = 0;
  for (size_t a = 0; a < x->ndim; ++a) {
    const bool is_reduced = ((axes_mask >> a) & 1u) != 0u;
    if (is_reduced && keepdims) {
      out_dims[out_ndim++] = 1;
    } else if (!is_reduced) {
      out_dims[out_ndim++] = x->shape[a];
    }
  }
  if (out_ndim == 0) {
    /* All axes reduced: a scalar stays a [1] tensor. */
    out_dims[out_ndim++] = 1;
  }
  FloatTensor *out = st_create(out_ndim, out_dims);
  if (out == NULL) {
    return NULL;
  }
  if (!st_reduce_max_into(x, axes_mask, keepdims, out)) {
    st_destroy(out);
    return NULL;
  }
  return out;
}

/* ---- Batched matmul ------------------------------------------------------ */

/* OpenMP distributes batch GEMMs only when there are many batches and each
 * product is small enough that the BLAS backend stays single-threaded;
 * bench_st_matmul measured the cutover (oversubscribing the BLAS threads
 * regresses up to 2x). */
#define ST_MATMUL_OMP_MIN_BATCHES 8u
#define ST_MATMUL_OMP_MAX_GEMM 262144u

/* One GEMM on raw buffers through the sm_gemm backend (like nm_lib's gemm). */
static bool st_matmul_gemm(const float *a, const float *b, float *c, size_t m,
                           size_t k, size_t n) {
  FloatMatrix ma = {.rows = m, .cols = k, .capacity = m * k,
                    .values = (float *)a};
  FloatMatrix mb = {.rows = k, .cols = n, .capacity = k * n,
                    .values = (float *)b};
  FloatMatrix mc = {.rows = m, .cols = n, .capacity = m * n, .values = c};
  return sm_gemm(&mc, 1.0f, &ma, SM_NO_TRANSPOSE, &mb, SM_NO_TRANSPOSE, 0.0f);
}

static bool st_matmul_batched_f32(const float *a, size_t a_ndim,
                                  const size_t *a_dims, const float *b,
                                  size_t b_ndim, const size_t *b_dims,
                                  float *c, size_t c_ndim,
                                  const size_t *c_dims) {
  const size_t m = a_dims[a_ndim - 2];
  const size_t k = a_dims[a_ndim - 1];
  const size_t n = b_dims[b_ndim - 1];

  size_t b_numel = 1;
  for (size_t i = 0; i < b_ndim; ++i) {
    b_numel *= b_dims[i];
  }
  if (b_numel == k * n) {
    /* b is one matrix shared by every batch: a's batches stack into the
     * rows of one product (nm_lib's projection fast path). */
    size_t a_numel = 1;
    for (size_t i = 0; i < a_ndim; ++i) {
      a_numel *= a_dims[i];
    }
    return st_matmul_gemm(a, b, c, a_numel / k, k, n);
  }

  /* Batch broadcast over the leading axes (nm_lib's padded decomposition). */
  const size_t batch_a = a_ndim - 2;
  const size_t batch_b = b_ndim - 2;
  const size_t batch_rank = batch_a > batch_b ? batch_a : batch_b;
  size_t da[ST_MAX_DIMS] = {0};
  size_t db[ST_MAX_DIMS] = {0};
  size_t d[ST_MAX_DIMS] = {0};
  for (size_t i = 0; i < batch_rank; ++i) {
    da[i] = i < batch_rank - batch_a ? 1 : a_dims[i - (batch_rank - batch_a)];
    db[i] = i < batch_rank - batch_b ? 1 : b_dims[i - (batch_rank - batch_b)];
    if (da[i] != db[i] && da[i] != 1 && db[i] != 1) {
      return false;
    }
    d[i] = da[i] == 1 ? db[i] : da[i];
  }
  size_t batches = 1;
  for (size_t i = 0; i < batch_rank; ++i) {
    batches *= d[i];
  }

  const size_t gemm_work = m * k * n;
  const bool use_omp =
      batches >= ST_MATMUL_OMP_MIN_BATCHES && gemm_work <= ST_MATMUL_OMP_MAX_GEMM &&
      omp_get_max_threads() > 1;
  if (use_omp) {
#pragma omp parallel for schedule(static)
    for (size_t index = 0; index < batches; ++index) {
      size_t rest = index;
      size_t offset_a = 0;
      size_t offset_b = 0;
      size_t scale_a = 1;
      size_t scale_b = 1;
      for (size_t i = batch_rank; i-- > 0;) {
        const size_t coordinate = rest % d[i];
        rest /= d[i];
        offset_a += (da[i] == 1 ? 0 : coordinate) * scale_a;
        offset_b += (db[i] == 1 ? 0 : coordinate) * scale_b;
        scale_a *= da[i];
        scale_b *= db[i];
      }
      st_matmul_gemm(a + offset_a * m * k, b + offset_b * k * n,
                     c + index * m * n, m, k, n);
    }
    return true;
  }
  for (size_t index = 0; index < batches; ++index) {
    size_t rest = index;
    size_t offset_a = 0;
    size_t offset_b = 0;
    size_t scale_a = 1;
    size_t scale_b = 1;
    for (size_t i = batch_rank; i-- > 0;) {
      const size_t coordinate = rest % d[i];
      rest /= d[i];
      offset_a += (da[i] == 1 ? 0 : coordinate) * scale_a;
      offset_b += (db[i] == 1 ? 0 : coordinate) * scale_b;
      scale_a *= da[i];
      scale_b *= db[i];
    }
    if (!st_matmul_gemm(a + offset_a * m * k, b + offset_b * k * n,
                        c + index * m * n, m, k, n)) {
      return false;
    }
  }
  return true;
}

bool st_matmul_batched(const FloatTensor *a, const FloatTensor *b,
                       FloatTensor *c) {
  if (!st_interp_tensor_valid(a) || !st_interp_tensor_valid(b) ||
      !st_interp_tensor_valid(c)) {
    ST_LOG_ERROR("st_matmul_batched: invalid tensor input.");
    return false;
  }
  ST_ASSERT_NOT_PENDING(a);
  ST_ASSERT_NOT_PENDING(b);
  ST_ASSERT_NOT_PENDING(c);
  if (a->ndim < 2 || b->ndim < 2) {
    ST_LOG_ERROR("st_matmul_batched: operands need rank >= 2.");
    return false;
  }
  const size_t m = a->shape[a->ndim - 2];
  const size_t k = a->shape[a->ndim - 1];
  const size_t n = b->shape[b->ndim - 1];
  if (b->shape[b->ndim - 2] != k) {
    ST_LOG_ERROR("st_matmul_batched: inner dimensions mismatch.");
    return false;
  }

  /* Expected output shape: broadcast batch shape plus [M, N]. */
  size_t out_dims[ST_MAX_DIMS];
  const size_t batch_a = a->ndim - 2;
  const size_t batch_b = b->ndim - 2;
  const size_t batch_rank = batch_a > batch_b ? batch_a : batch_b;
  if (batch_rank + 2 > ST_MAX_DIMS) {
    return false;
  }
  for (size_t i = 0; i < batch_rank; ++i) {
    const size_t da = i < batch_rank - batch_a ? 1 : a->shape[i - (batch_rank - batch_a)];
    const size_t db = i < batch_rank - batch_b ? 1 : b->shape[i - (batch_rank - batch_b)];
    if (da != db && da != 1 && db != 1) {
      ST_LOG_ERROR("st_matmul_batched: batch shapes do not broadcast.");
      return false;
    }
    out_dims[i] = da == 1 ? db : da;
  }
  out_dims[batch_rank] = m;
  out_dims[batch_rank + 1] = n;
  const size_t out_ndim = batch_rank + 2;
  if (c->ndim != out_ndim ||
      memcmp(c->shape, out_dims, out_ndim * sizeof(size_t)) != 0) {
    ST_LOG_ERROR("st_matmul_batched: output shape mismatch.");
    return false;
  }

  /* bf16 promotes to f32; a bf16 output is rounded back at the end. */
  FloatTensor *a_f32 = a->dtype == ST_DTYPE_BF16 ? st_to_f32(a) : NULL;
  FloatTensor *b_f32 = b->dtype == ST_DTYPE_BF16 ? st_to_f32(b) : NULL;
  if ((a->dtype == ST_DTYPE_BF16 && a_f32 == NULL) ||
      (b->dtype == ST_DTYPE_BF16 && b_f32 == NULL)) {
    st_destroy(b_f32);
    st_destroy(a_f32);
    return false;
  }
  const float *a_values = a_f32 ? a_f32->values : a->values;
  const float *b_values = b_f32 ? b_f32->values : b->values;

  FloatTensor *c_f32 = NULL;
  float *c_values = c->values;
  if (c->dtype == ST_DTYPE_BF16) {
    c_f32 = st_create(c->ndim, c->shape);
    if (c_f32 == NULL) {
      st_destroy(b_f32);
      st_destroy(a_f32);
      return false;
    }
    c_values = c_f32->values;
  }

  const bool ok = st_matmul_batched_f32(a_values, a->ndim, a->shape, b_values,
                                        b->ndim, b->shape, c_values, c->ndim,
                                        c->shape);
  if (ok && c_f32 != NULL) {
    st_f32_to_bf16_bulk(c_f32->values, (uint16_t *)c->values, c->numel);
  }
  st_destroy(c_f32);
  st_destroy(b_f32);
  st_destroy(a_f32);
  return ok;
}

/* ---- Gather --------------------------------------------------------------- */

/* OpenMP over the outer x count copy blocks only above this many copied
 * elements: bench_st_gather measured a 2x regression at 1M copied elements
 * (memcpy-bound work plus per-block index math), so realistic gathers stay
 * sequential. */
#define ST_GATHER_OMP_THRESHOLD 4194304u

bool st_gather_raw(const void *table, size_t table_ndim,
                   const size_t *table_shape, const int64_t *indices,
                   size_t count, size_t axis, size_t elem_bytes, void *out) {
  if (table == NULL || table_shape == NULL || indices == NULL || out == NULL ||
      table_ndim == 0 || table_ndim > ST_MAX_DIMS || count == 0 ||
      axis >= table_ndim || elem_bytes == 0) {
    ST_LOG_ERROR("st_gather_raw: invalid input.");
    return false;
  }
  size_t outer = 1;
  size_t dim = table_shape[axis];
  size_t inner = 1;
  for (size_t i = 0; i < table_ndim; ++i) {
    if (table_shape[i] == 0) {
      ST_LOG_ERROR("st_gather_raw: invalid shape.");
      return false;
    }
    if (i < axis) {
      outer *= table_shape[i];
    } else if (i > axis) {
      inner *= table_shape[i];
    }
  }
  const size_t bytes = inner * elem_bytes;
  const size_t row_bytes = dim * bytes;
  const size_t total = count * bytes;
  if (row_bytes > SIZE_MAX / outer || total > SIZE_MAX / (outer * dim)) {
    return false;
  }

  /* Validate every index before writing anything, so an out-of-range index
   * fails without partial output (nm_lib copies until it hits the bad one). */
  for (size_t j = 0; j < count; ++j) {
    const int64_t index = indices[j] < 0 ? indices[j] + (int64_t)dim : indices[j];
    if (index < 0 || index >= (int64_t)dim) {
      ST_LOG_ERROR("st_gather_raw: index out of range.");
      return false;
    }
  }

  const uint8_t *src = table;
  uint8_t *dst = out;
  const size_t blocks = outer * count;
  const size_t copied = blocks * bytes;
  if (copied > ST_GATHER_OMP_THRESHOLD && omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
    for (size_t block = 0; block < blocks; ++block) {
      const size_t o = block / count;
      const size_t j = block % count;
      const int64_t index =
          indices[j] < 0 ? indices[j] + (int64_t)dim : indices[j];
      memcpy(dst + block * bytes,
             src + (o * dim + (size_t)index) * bytes, bytes);
    }
    return true;
  }
  for (size_t o = 0; o < outer; ++o) {
    for (size_t j = 0; j < count; ++j) {
      const int64_t index =
          indices[j] < 0 ? indices[j] + (int64_t)dim : indices[j];
      memcpy(dst, src + (o * dim + (size_t)index) * bytes, bytes);
      dst += bytes;
    }
  }
  return true;
}

bool st_gather(const FloatTensor *table, const int64_t *indices, size_t count,
               size_t axis, FloatTensor *out) {
  if (!st_interp_tensor_valid(table) || !st_interp_tensor_valid(out)) {
    ST_LOG_ERROR("st_gather: invalid tensor input.");
    return false;
  }
  ST_ASSERT_NOT_PENDING(table);
  ST_ASSERT_NOT_PENDING(out);
  if (axis >= table->ndim) {
    ST_LOG_ERROR("st_gather: axis out of range.");
    return false;
  }
  if (table->dtype != out->dtype) {
    ST_LOG_ERROR("st_gather: dtype mismatch.");
    return false;
  }
  if (out->ndim != table->ndim ||
      memcmp(out->shape, table->shape, axis * sizeof(size_t)) != 0 ||
      out->shape[axis] != count ||
      memcmp(out->shape + axis + 1, table->shape + axis + 1,
             (table->ndim - axis - 1) * sizeof(size_t)) != 0) {
    ST_LOG_ERROR("st_gather: output shape mismatch.");
    return false;
  }
  return st_gather_raw(table->values, table->ndim, table->shape, indices,
                      count, axis, st_dtype_size(table->dtype), out->values);
}

/* ---- Materializing permute ------------------------------------------------ */

bool st_permute_copy_into_bytes(const void *src, size_t ndim,
                                const size_t *shape, const size_t *perm,
                                size_t elem_bytes, void *out) {
  if (src == NULL || shape == NULL || perm == NULL || out == NULL ||
      ndim == 0 || ndim > ST_MAX_DIMS || elem_bytes == 0) {
    ST_LOG_ERROR("st_permute_copy_into_bytes: invalid input.");
    return false;
  }
  bool seen[ST_MAX_DIMS] = {false};
  size_t in_stride[ST_MAX_DIMS];
  size_t running = 1;
  for (size_t i = ndim; i-- > 0;) {
    if (shape[i] == 0) {
      ST_LOG_ERROR("st_permute_copy_into_bytes: invalid shape.");
      return false;
    }
    in_stride[i] = running;
    running *= shape[i];
  }
  for (size_t i = 0; i < ndim; ++i) {
    if (perm[i] >= ndim || seen[perm[i]]) {
      ST_LOG_ERROR("st_permute_copy_into_bytes: perm is not a permutation.");
      return false;
    }
    seen[perm[i]] = true;
  }

  /* Walk the output space (the permuted shape); output axis i steps
   * through the input with in_stride[perm[i]] (nm_lib's run_transpose +
   * walk_copy). */
  size_t out_shape[ST_MAX_DIMS];
  for (size_t i = 0; i < ndim; ++i) {
    out_shape[i] = shape[perm[i]];
  }
  StWalk w;
  st_walk_init(&w, ndim, out_shape, 1);
  {
    const size_t offset = ST_WALK_DIMS - ndim;
    for (size_t i = 0; i < ndim; ++i) {
      w.stride[0][offset + i] = (ptrdiff_t)in_stride[perm[i]];
    }
  }
  st_walk_finish(&w);
  const size_t inner = st_walk_inner(&w);
  const size_t rows = st_walk_rows(&w);
  const ptrdiff_t step = w.stride[0][w.rank - 1];
  uint8_t *dst = out;
  for (size_t r = 0; r < rows; ++r, st_walk_next_row(&w), dst += inner * elem_bytes) {
    const uint8_t *from = (const uint8_t *)src +
                           (size_t)w.offset[0] * elem_bytes;
    if (step == 1) {
      memcpy(dst, from, inner * elem_bytes);
    } else if (elem_bytes == 4) {
      const float *from_f = (const float *)(const void *)from;
      float *to_f = (float *)(void *)dst;
      for (size_t i = 0; i < inner; ++i) {
        to_f[i] = from_f[(ptrdiff_t)i * step];
      }
    } else if (elem_bytes == 8) {
      const int64_t *from_i = (const int64_t *)(const void *)from;
      int64_t *to_i = (int64_t *)(void *)dst;
      for (size_t i = 0; i < inner; ++i) {
        to_i[i] = from_i[(ptrdiff_t)i * step];
      }
    } else {
      for (size_t i = 0; i < inner; ++i) {
        memcpy(dst + i * elem_bytes,
               from + (size_t)((ptrdiff_t)i * step) * elem_bytes, elem_bytes);
      }
    }
  }
  return true;
}

bool st_permute_copy_into(const FloatTensor *src, const size_t *perm,
                          FloatTensor *out) {
  if (!st_interp_tensor_valid(src) || !st_interp_tensor_valid(out)) {
    ST_LOG_ERROR("st_permute_copy_into: invalid tensor input.");
    return false;
  }
  ST_ASSERT_NOT_PENDING(src);
  ST_ASSERT_NOT_PENDING(out);
  if (perm == NULL || src->dtype != out->dtype || out->ndim != src->ndim) {
    ST_LOG_ERROR("st_permute_copy_into: invalid input.");
    return false;
  }
  for (size_t i = 0; i < src->ndim; ++i) {
    if (out->shape[i] != src->shape[perm[i]]) {
      ST_LOG_ERROR("st_permute_copy_into: output shape mismatch.");
      return false;
    }
  }
  return st_permute_copy_into_bytes(src->values, src->ndim, src->shape, perm,
                                    st_dtype_size(src->dtype), out->values);
}
