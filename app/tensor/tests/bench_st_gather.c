/*
 * Median benchmark for gather and materializing permute: nm_lib's
 * run_gather/run_transpose loops (before) versus the st_interp kernels.
 * Run: bazel run //app/tensor:bench_st_gather --config=benchmark
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
  if (n <= 65536) return 400;
  if (n <= 1048576) return 60;
  return 12;
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

/* nm_lib's run_gather (without its partial-write failure behavior). */
static void nm_run_gather(const void *data, size_t ndim, const size_t *dims,
                          size_t axis, const int64_t *indices, size_t count,
                          size_t elem, void *out) {
  size_t outer = 1;
  size_t inner = 1;
  for (size_t i = 0; i < axis; ++i) {
    outer *= dims[i];
  }
  for (size_t i = axis + 1; i < ndim; ++i) {
    inner *= dims[i];
  }
  const int64_t dim = (int64_t)dims[axis];
  const size_t bytes = inner * elem;
  uint8_t *dst = out;
  for (size_t o = 0; o < outer; ++o) {
    for (size_t j = 0; j < count; ++j) {
      const int64_t index = indices[j] < 0 ? indices[j] + dim : indices[j];
      memcpy(dst,
             (const uint8_t *)data + (o * (size_t)dim + (size_t)index) * bytes,
             bytes);
      dst += bytes;
    }
  }
}

/* nm_lib's run_transpose + walk_copy. */
static void nm_run_transpose(const void *in, size_t ndim, const size_t *dims,
                             const size_t *perm, size_t elem, void *out) {
  size_t out_dims[BENCH_DIMS];
  size_t in_stride[BENCH_DIMS];
  size_t running = 1;
  for (size_t i = ndim; i-- > 0;) {
    in_stride[i] = running;
    running *= dims[i];
  }
  for (size_t i = 0; i < ndim; ++i) {
    out_dims[i] = dims[perm[i]];
  }
  /* Walk rows of the output space. */
  size_t d[BENCH_DIMS];
  size_t c[BENCH_DIMS] = {0};
  ptrdiff_t stride[BENCH_DIMS];
  ptrdiff_t offset = 0;
  const size_t pad = BENCH_DIMS - ndim;
  for (size_t i = 0; i < BENCH_DIMS; ++i) {
    d[i] = i < pad ? 1 : out_dims[i - pad];
  }
  for (size_t i = 0; i < BENCH_DIMS; ++i) {
    stride[i] = i < pad ? 0 : (ptrdiff_t)in_stride[perm[i - pad]];
  }
  size_t rank = 0;
  size_t wd[BENCH_DIMS];
  ptrdiff_t wstride[BENCH_DIMS];
  for (size_t i = 0; i < BENCH_DIMS; ++i) {
    if (d[i] == 1) {
      continue;
    }
    const bool merge =
        rank > 0 && wstride[rank - 1] == stride[i] * (ptrdiff_t)d[i];
    if (merge) {
      wd[rank - 1] *= d[i];
      wstride[rank - 1] = stride[i];
      continue;
    }
    wd[rank] = d[i];
    wstride[rank] = stride[i];
    rank++;
  }
  if (rank == 0) {
    wd[0] = 1;
    wstride[0] = 0;
    rank = 1;
  }
  const size_t inner = wd[rank - 1];
  size_t rows = 1;
  for (size_t i = 0; i + 1 < rank; ++i) {
    rows *= wd[i];
  }
  const ptrdiff_t step = wstride[rank - 1];
  uint8_t *dst = out;
  for (size_t r = 0; r < rows; ++r) {
    const uint8_t *from = (const uint8_t *)in + (size_t)offset * elem;
    if (step == 1) {
      memcpy(dst, from, inner * elem);
    } else {
      for (size_t i = 0; i < inner; ++i) {
        memcpy(dst + i * elem,
               from + (size_t)((ptrdiff_t)i * step) * elem, elem);
      }
    }
    dst += inner * elem;
    for (size_t k = rank - 1; k-- > 0;) {
      offset += wstride[k];
      if (++c[k] < wd[k]) {
        break;
      }
      offset -= wstride[k] * (ptrdiff_t)wd[k];
      c[k] = 0;
    }
  }
}

static void measure(const char *name, size_t elements, void (*fn)(void *),
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
  printf("| %s | %s | %.3f | %.3f | %.2fx |\n", name,
         omp_get_max_threads() == 1 ? "1" : "default", b, a, b / a);
  (void)fn;
}

typedef struct {
  FloatTensor *table;
  FloatTensor *out;
  int64_t *indices;
  size_t count;
  size_t axis;
} GatherCtx;

static void gather_before(void *ctx) {
  GatherCtx *c = ctx;
  nm_run_gather(st_tensor_data(c->table), c->table->ndim, c->table->shape,
                c->axis, c->indices, c->count, sizeof(float),
                st_tensor_mutable_data(c->out));
}

static void gather_after(void *ctx) {
  GatherCtx *c = ctx;
  if (!st_gather(c->table, c->indices, c->count, c->axis, c->out)) {
    fprintf(stderr, "st_gather failed\n");
    exit(1);
  }
}

static void run_gather_case(const char *name, size_t ndim, const size_t *dims,
                            size_t axis, size_t count) {
  GatherCtx ctx;
  ctx.table = make_input(ndim, dims);
  ctx.count = count;
  ctx.axis = axis;
  ctx.indices = malloc(count * sizeof(int64_t));
  if (ctx.indices == NULL) {
    exit(1);
  }
  for (size_t i = 0; i < count; ++i) {
    ctx.indices[i] = (int64_t)((i * 7919) % dims[axis]);
  }
  size_t out_dims[ST_MAX_DIMS];
  memcpy(out_dims, dims, ndim * sizeof(size_t));
  out_dims[axis] = count;
  ctx.out = st_create(ndim, out_dims);
  if (ctx.out == NULL) {
    exit(1);
  }
  const size_t elements = st_tensor_numel(ctx.out);
  measure(name, elements, NULL, gather_before, gather_after, &ctx);
  st_destroy(ctx.out);
  free(ctx.indices);
  st_destroy(ctx.table);
}

typedef struct {
  FloatTensor *src;
  FloatTensor *out;
  const size_t *perm;
} PermuteCtx;

static void permute_before(void *ctx) {
  PermuteCtx *c = ctx;
  nm_run_transpose(st_tensor_data(c->src), c->src->ndim, c->src->shape,
                   c->perm, sizeof(float), st_tensor_mutable_data(c->out));
}

static void permute_after(void *ctx) {
  PermuteCtx *c = ctx;
  if (!st_permute_copy_into(c->src, c->perm, c->out)) {
    fprintf(stderr, "st_permute_copy_into failed\n");
    exit(1);
  }
}

static void run_permute_case(const char *name, size_t ndim,
                             const size_t *dims, const size_t *perm) {
  PermuteCtx ctx;
  ctx.src = make_input(ndim, dims);
  ctx.perm = perm;
  size_t out_dims[ST_MAX_DIMS];
  for (size_t i = 0; i < ndim; ++i) {
    out_dims[i] = dims[perm[i]];
  }
  ctx.out = st_create(ndim, out_dims);
  if (ctx.out == NULL) {
    exit(1);
  }
  measure(name, st_tensor_numel(ctx.src), NULL, permute_before, permute_after,
          &ctx);
  st_destroy(ctx.out);
  st_destroy(ctx.src);
}

int main(void) {
  printf("| Case | Threads | nm_lib before (µs) | st kernel (µs) | Speedup |\n");
  printf("| --- | --- | ---: | ---: | ---: |\n");
  const int default_threads = omp_get_max_threads();

  const size_t perm_attn[4] = {0, 2, 1, 3};
  const size_t perm_last[4] = {0, 1, 3, 2};
  const size_t perm_small[4] = {0, 2, 1, 3};

  omp_set_num_threads(default_threads);
  run_gather_case("gather [30000,256] axis 0, count 128", 2,
                  (size_t[2]){30000, 256}, 0, 128);
  run_gather_case("gather [30000,64] axis 0, count 16", 2,
                  (size_t[2]){30000, 64}, 0, 16);
  run_gather_case("gather [2,30000,1024] axis 1, count 512", 3,
                  (size_t[3]){2, 30000, 1024}, 1, 512);
  run_gather_case("gather [100,7,3] axis 1, count 7", 3,
                  (size_t[3]){100, 7, 3}, 1, 7);
  run_permute_case("permute [2,4,128,64] -> [2,128,4,64]", 4,
                   (size_t[4]){2, 4, 128, 64}, perm_attn);
  run_permute_case("permute [2,4,128,64] -> [2,4,64,128]", 4,
                   (size_t[4]){2, 4, 128, 64}, perm_last);
  run_permute_case("permute [1,4,16,16] -> [1,16,4,16]", 4,
                   (size_t[4]){1, 4, 16, 16}, perm_small);

  omp_set_num_threads(1);
  run_gather_case("gather [30000,256] axis 0, count 128", 2,
                  (size_t[2]){30000, 256}, 0, 128);
  run_gather_case("gather [2,30000,1024] axis 1, count 512", 3,
                  (size_t[3]){2, 30000, 1024}, 1, 512);
  run_permute_case("permute [2,4,128,64] -> [2,128,4,64]", 4,
                   (size_t[4]){2, 4, 128, 64}, perm_attn);
  return 0;
}
