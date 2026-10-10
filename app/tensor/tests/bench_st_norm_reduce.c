/*
 * Median benchmark for LayerNorm and ReduceMax: nm_lib's interpreter loops
 * (before) versus the st_interp kernels (after).
 * Run: bazel run //app/tensor:bench_st_norm_reduce --config=benchmark
 */

#include "st.h"
#include "st_interp.h"

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BENCH_DIMS 8

/* ---- Timing helpers ----------------------------------------------------- */

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double median(double *samples, size_t count) {
  for (size_t i = 1; i < count; ++i) {
    const double value = samples[i];
    size_t j = i;
    while (j > 0 && samples[j - 1] > value) {
      samples[j] = samples[j - 1];
      --j;
    }
    samples[j] = value;
  }
  return samples[count / 2];
}

static size_t repetitions_for(size_t n) {
  if (n <= 4096) return 2000;
  if (n <= 65536) return 400;
  if (n <= 1048576) return 40;
  return 8;
}

static FloatTensor *make_input(size_t ndim, const size_t *shape) {
  FloatTensor *t = st_create(ndim, shape);
  if (t == NULL) {
    exit(1);
  }
  float *values = st_tensor_mutable_data(t);
  for (size_t i = 0; i < st_tensor_numel(t); ++i) {
    values[i] = sinf((float)i * 0.7351f) * 1.5f + 0.5f;
  }
  return t;
}

static void measure(const char *name, size_t elements, size_t out_elements,
                    void (*before)(void *), void (*after)(void *),
                    void *context) {
  const size_t reps = repetitions_for(elements);
  double before_samples[9];
  double after_samples[9];
  for (size_t s = 0; s < 9; ++s) {
    uint64_t start = now_ns();
    for (size_t r = 0; r < reps; ++r) {
      before(context);
    }
    before_samples[s] = (double)(now_ns() - start) / (double)reps / 1000.0;

    start = now_ns();
    for (size_t r = 0; r < reps; ++r) {
      after(context);
    }
    after_samples[s] = (double)(now_ns() - start) / (double)reps / 1000.0;
  }
  const double b = median(before_samples, 9);
  const double a = median(after_samples, 9);
  printf("| %s | %s | %zu | %.3f | %.3f | %.2fx |\n", name,
         omp_get_max_threads() == 1 ? "1" : "default", elements, b, a, b / a);
  (void)out_elements;
}

/* ---- nm_lib references --------------------------------------------------- */

/* nm_lib's sum8 (nm_graph_interp.c). */
static float nm_sum8(const float *x, size_t n) {
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

/* nm_lib's run_layernorm. */
static void nm_run_layernorm(const float *in, size_t ndim,
                             const size_t *dims, size_t norm_rank,
                             const float *scale, const float *bias,
                             float eps, float *out) {
  size_t group = 1;
  for (size_t i = ndim - norm_rank; i < ndim; ++i) {
    group *= dims[i];
  }
  size_t numel = 1;
  for (size_t i = 0; i < ndim; ++i) {
    numel *= dims[i];
  }
  const size_t groups = numel / group;
  for (size_t g = 0; g < groups; ++g) {
    const float *x = in + g * group;
    float *y = out + g * group;
    const float mean = nm_sum8(x, group) / (float)group;
    for (size_t j = 0; j < group; ++j) {
      y[j] = x[j] - mean;
    }
    float lanes[8] = {0.0f};
    size_t j = 0;
    for (; j + 8 <= group; j += 8) {
      for (size_t l = 0; l < 8; ++l) {
        lanes[l] += y[j + l] * y[j + l];
      }
    }
    float variance = 0.0f;
    for (; j < group; ++j) {
      variance += y[j] * y[j];
    }
    for (size_t l = 0; l < 8; ++l) {
      variance += lanes[l];
    }
    variance /= (float)group;
    const float inverse = 1.0f / sqrtf(variance + eps);
    if (scale && bias) {
      for (size_t i = 0; i < group; ++i) {
        y[i] = y[i] * inverse * scale[i] + bias[i];
      }
    } else {
      for (size_t i = 0; i < group; ++i) {
        float value = y[i] * inverse;
        if (scale) {
          value *= scale[i];
        }
        if (bias) {
          value += bias[i];
        }
        y[i] = value;
      }
    }
  }
}

/* nm_lib's run_reducemax: walk the input, the output offset stands still
 * along reduced axes. */
typedef struct {
  size_t d[BENCH_DIMS];
  size_t c[BENCH_DIMS];
  ptrdiff_t stride[2][BENCH_DIMS];
  ptrdiff_t offset[2];
  size_t rank;
} NmWalk;

static void nm_pad_dims(size_t ndim, const size_t *shape, size_t out[BENCH_DIMS]) {
  const size_t offset = BENCH_DIMS - ndim;
  for (size_t i = 0; i < BENCH_DIMS; ++i) {
    out[i] = i < offset ? 1 : shape[i - offset];
  }
}

static void nm_broadcast_strides(size_t ndim, const size_t *shape,
                                 const size_t d[BENCH_DIMS],
                                 ptrdiff_t out[BENCH_DIMS]) {
  size_t padded[BENCH_DIMS];
  nm_pad_dims(ndim, shape, padded);
  ptrdiff_t running = 1;
  for (size_t i = BENCH_DIMS; i-- > 0;) {
    out[i] = padded[i] == 1 && d[i] != 1 ? 0 : running;
    running *= (ptrdiff_t)padded[i];
  }
}

static void nm_walk_init(NmWalk *w, size_t ndim, const size_t *shape) {
  memset(w, 0, sizeof(*w));
  nm_pad_dims(ndim, shape, w->d);
}

static void nm_walk_finish(NmWalk *w) {
  size_t rank = 0;
  for (size_t i = 0; i < BENCH_DIMS; ++i) {
    if (w->d[i] == 1) {
      continue;
    }
    const bool merge =
        rank > 0 && w->stride[0][rank - 1] ==
                        w->stride[0][i] * (ptrdiff_t)w->d[i] &&
        w->stride[1][rank - 1] == w->stride[1][i] * (ptrdiff_t)w->d[i];
    if (merge) {
      w->d[rank - 1] *= w->d[i];
      w->stride[0][rank - 1] = w->stride[0][i];
      w->stride[1][rank - 1] = w->stride[1][i];
      continue;
    }
    w->d[rank] = w->d[i];
    w->stride[0][rank] = w->stride[0][i];
    w->stride[1][rank] = w->stride[1][i];
    rank++;
  }
  if (rank == 0) {
    w->d[0] = 1;
    w->stride[0][0] = 0;
    w->stride[1][0] = 0;
    rank = 1;
  }
  w->rank = rank;
}

static void nm_walk_next_row(NmWalk *w) {
  for (size_t k = w->rank - 1; k-- > 0;) {
    w->offset[0] += w->stride[0][k];
    w->offset[1] += w->stride[1][k];
    if (++w->c[k] < w->d[k]) {
      return;
    }
    w->offset[0] -= w->stride[0][k] * (ptrdiff_t)w->d[k];
    w->offset[1] -= w->stride[1][k] * (ptrdiff_t)w->d[k];
    w->c[k] = 0;
  }
}

static void nm_run_reducemax(const float *in, size_t ndim,
                             const size_t *dims, uint32_t mask, float *out,
                             size_t out_numel) {
  size_t kept[BENCH_DIMS];
  for (size_t i = 0; i < ndim; ++i) {
    kept[i] = (mask >> i) & 1u ? 1 : dims[i];
  }
  NmWalk w;
  nm_walk_init(&w, ndim, dims);
  nm_broadcast_strides(ndim, kept, w.d, w.stride[0]);
  nm_broadcast_strides(ndim, dims, w.d, w.stride[1]);
  nm_walk_finish(&w);
  for (size_t i = 0; i < out_numel; ++i) {
    out[i] = -INFINITY;
  }
  const size_t inner = w.d[w.rank - 1];
  size_t rows = 1;
  for (size_t i = 0; i + 1 < w.rank; ++i) {
    rows *= w.d[i];
  }
  const ptrdiff_t to = w.stride[0][w.rank - 1];
  for (size_t r = 0; r < rows; ++r, nm_walk_next_row(&w)) {
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

/* ---- Benchmark contexts -------------------------------------------------- */

typedef struct {
  FloatTensor *x;
  FloatTensor *gamma;
  FloatTensor *beta;
  FloatTensor *out;
  size_t norm_rank;
  float eps;
} LayernormCtx;

static void layernorm_before(void *ctx) {
  LayernormCtx *c = ctx;
  nm_run_layernorm(st_tensor_data(c->x), c->x->ndim, c->x->shape,
                   c->norm_rank, st_tensor_data(c->gamma),
                   st_tensor_data(c->beta), c->eps,
                   st_tensor_mutable_data(c->out));
}

static void layernorm_after(void *ctx) {
  LayernormCtx *c = ctx;
  if (!st_layernorm(c->x, c->gamma, c->beta, c->norm_rank, c->eps, c->out)) {
    fprintf(stderr, "st_layernorm failed\n");
    exit(1);
  }
}

typedef struct {
  FloatTensor *x;
  FloatTensor *out;
  uint32_t mask;
  bool keepdims;
} ReduceCtx;

static void reduce_before(void *ctx) {
  ReduceCtx *c = ctx;
  nm_run_reducemax(st_tensor_data(c->x), c->x->ndim, c->x->shape, c->mask,
                   st_tensor_mutable_data(c->out), c->out->numel);
}

static void reduce_after(void *ctx) {
  ReduceCtx *c = ctx;
  if (!st_reduce_max_into(c->x, c->mask, c->keepdims, c->out)) {
    fprintf(stderr, "st_reduce_max_into failed\n");
    exit(1);
  }
}

static void run_layernorm_case(const char *name, size_t ndim,
                               const size_t *dims, size_t norm_rank) {
  size_t elements = 1;
  for (size_t i = 0; i < ndim; ++i) {
    elements *= dims[i];
  }
  size_t group = 1;
  for (size_t i = ndim - norm_rank; i < ndim; ++i) {
    group *= dims[i];
  }
  LayernormCtx ctx;
  ctx.x = make_input(ndim, dims);
  ctx.gamma = make_input(1, (size_t[1]){group});
  ctx.beta = make_input(1, (size_t[1]){group});
  ctx.out = st_create(ndim, dims);
  ctx.norm_rank = norm_rank;
  ctx.eps = 1e-5f;
  if (ctx.out == NULL) {
    exit(1);
  }
  measure(name, elements, elements, layernorm_before, layernorm_after, &ctx);
  st_destroy(ctx.out);
  st_destroy(ctx.beta);
  st_destroy(ctx.gamma);
  st_destroy(ctx.x);
}

static void run_reduce_case(const char *name, size_t ndim, const size_t *dims,
                            uint32_t mask) {
  size_t elements = 1;
  size_t out_elements = 1;
  for (size_t i = 0; i < ndim; ++i) {
    elements *= dims[i];
    if (!((mask >> i) & 1u)) {
      out_elements *= dims[i];
    }
  }
  size_t out_dims[ST_MAX_DIMS];
  size_t out_ndim = 0;
  for (size_t i = 0; i < ndim; ++i) {
    if ((mask >> i) & 1u) {
      out_dims[out_ndim++] = 1;
    } else {
      out_dims[out_ndim++] = dims[i];
    }
  }
  ReduceCtx ctx;
  ctx.x = make_input(ndim, dims);
  ctx.out = st_create(out_ndim, out_dims);
  ctx.mask = mask;
  ctx.keepdims = true;
  if (ctx.out == NULL) {
    exit(1);
  }
  measure(name, elements, out_elements, reduce_before, reduce_after, &ctx);
  st_destroy(ctx.out);
  st_destroy(ctx.x);
}

int main(void) {
  printf("| Case | Threads | Elements | nm_lib before (µs) | st kernel (µs) | Speedup |\n");
  printf("| --- | --- | ---: | ---: | ---: | ---: |\n");
  const int default_threads = omp_get_max_threads();

  omp_set_num_threads(default_threads);
  run_layernorm_case("layernorm [1,16,64]", 3, (size_t[3]){1, 16, 64}, 1);
  run_layernorm_case("layernorm [2,256,256]", 3, (size_t[3]){2, 256, 256}, 1);
  run_layernorm_case("layernorm [4,256,256]", 3, (size_t[3]){4, 256, 256}, 1);
  run_layernorm_case("layernorm [4,128,256]", 3, (size_t[3]){4, 128, 256}, 1);
  run_layernorm_case("layernorm [8,256,512]", 3, (size_t[3]){8, 256, 512}, 1);
  run_layernorm_case("layernorm [4,512,768]", 3, (size_t[3]){4, 512, 768}, 1);
  run_layernorm_case("layernorm [16,65,32] rank 2", 3,
                     (size_t[3]){16, 65, 32}, 2);
  run_reduce_case("reduce max [2,4,128,128] axis 3", 4,
                  (size_t[4]){2, 4, 128, 128}, 0x8);
  run_reduce_case("reduce max [1,4,16,16] axis 3", 4,
                  (size_t[4]){1, 4, 16, 16}, 0x8);
  run_reduce_case("reduce max [1,4,128,128] axis 3", 4,
                  (size_t[4]){1, 4, 128, 128}, 0x8);
  run_reduce_case("reduce max [4,128,256] axis 2", 3,
                  (size_t[3]){4, 128, 256}, 0x4);
  run_reduce_case("reduce max [4,128,256] axis 1", 3,
                  (size_t[3]){4, 128, 256}, 0x2);
  run_reduce_case("reduce max [2,1000] axis 0", 2, (size_t[2]){2, 1000}, 0x1);

  omp_set_num_threads(1);
  run_layernorm_case("layernorm [1,16,64]", 3, (size_t[3]){1, 16, 64}, 1);
  run_layernorm_case("layernorm [2,256,256]", 3, (size_t[3]){2, 256, 256}, 1);
  run_layernorm_case("layernorm [4,256,256]", 3, (size_t[3]){4, 256, 256}, 1);
  run_layernorm_case("layernorm [4,128,256]", 3, (size_t[3]){4, 128, 256}, 1);
  run_layernorm_case("layernorm [8,256,512]", 3, (size_t[3]){8, 256, 512}, 1);
  run_layernorm_case("layernorm [4,512,768]", 3, (size_t[3]){4, 512, 768}, 1);
  run_reduce_case("reduce max [2,4,128,128] axis 3", 4,
                  (size_t[4]){2, 4, 128, 128}, 0x8);
  run_reduce_case("reduce max [1,4,128,128] axis 3", 4,
                  (size_t[4]){1, 4, 128, 128}, 0x8);
  run_reduce_case("reduce max [4,128,256] axis 2", 3,
                  (size_t[3]){4, 128, 256}, 0x4);
  run_reduce_case("reduce max [4,128,256] axis 1", 3,
                  (size_t[3]){4, 128, 256}, 0x2);
  return 0;
}
