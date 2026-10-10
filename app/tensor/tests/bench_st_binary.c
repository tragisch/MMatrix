/*
 * Median benchmark for broadcast binary ops: nm_lib's run_binary loops
 * (before) versus the st_binary_broadcast kernels (after).
 * Run: bazel run //app/tensor:bench_st_binary --config=benchmark
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

/* ---- nm_lib reference (nm_graph_interp.c, run_binary) ------------------- */

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
    bool merge = rank > 0 && w->stride[0][rank - 1] ==
                                w->stride[0][i] * (ptrdiff_t)w->d[i] &&
                 w->stride[1][rank - 1] ==
                     w->stride[1][i] * (ptrdiff_t)w->d[i];
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

static float nm_apply_binary(StBinOp op, float a, float b) {
  switch (op) {
    case ST_BIN_ADD:
      return a + b;
    case ST_BIN_SUB:
      return a - b;
    case ST_BIN_MUL:
      return a * b;
    default:
      return a / b;
  }
}

static void nm_binary_row(StBinOp op, const float *a, const float *b,
                          float *out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = nm_apply_binary(op, a[i], b[i]);
  }
}

static void nm_binary_row_scalar_b(StBinOp op, const float *a, const float *b,
                                   float *out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = nm_apply_binary(op, a[i], b[0]);
  }
}

static void nm_binary_row_scalar_a(StBinOp op, const float *a, const float *b,
                                   float *out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = nm_apply_binary(op, a[0], b[i]);
  }
}

static void nm_run_binary(StBinOp op, const float *a, size_t a_ndim,
                          const size_t *a_dims, const float *b, size_t b_ndim,
                          const size_t *b_dims, float *out, size_t o_ndim,
                          const size_t *o_dims) {
  size_t n = 1, na = 1, nb = 1;
  for (size_t i = 0; i < o_ndim; ++i) n *= o_dims[i];
  for (size_t i = 0; i < a_ndim; ++i) na *= a_dims[i];
  for (size_t i = 0; i < b_ndim; ++i) nb *= b_dims[i];
  if (na == n && nb == n) {
    nm_binary_row(op, a, b, out, n);
    return;
  }
  if (na == n && nb == 1) {
    nm_binary_row_scalar_b(op, a, b, out, n);
    return;
  }
  if (na == n && nb > 0 && n % nb == 0 && b_ndim <= o_ndim) {
    bool suffix = true;
    for (size_t i = 0; i < b_ndim; ++i) {
      suffix = suffix && b_dims[i] == o_dims[o_ndim - b_ndim + i];
    }
    if (suffix) {
      for (size_t i = 0; i < n; i += nb) {
        nm_binary_row(op, a + i, b, out + i, nb);
      }
      return;
    }
  }
  NmWalk w;
  nm_walk_init(&w, o_ndim, o_dims);
  nm_broadcast_strides(a_ndim, a_dims, w.d, w.stride[0]);
  nm_broadcast_strides(b_ndim, b_dims, w.d, w.stride[1]);
  nm_walk_finish(&w);
  const size_t inner = w.d[w.rank - 1];
  size_t rows = 1;
  for (size_t i = 0; i + 1 < w.rank; ++i) {
    rows *= w.d[i];
  }
  const ptrdiff_t ta = w.stride[0][w.rank - 1];
  const ptrdiff_t tb = w.stride[1][w.rank - 1];
  for (size_t r = 0; r < rows; ++r, nm_walk_next_row(&w), out += inner) {
    const float *pa = a + w.offset[0];
    const float *pb = b + w.offset[1];
    if (ta == 1 && tb == 1) {
      nm_binary_row(op, pa, pb, out, inner);
    } else if (ta == 1 && tb == 0) {
      nm_binary_row_scalar_b(op, pa, pb, out, inner);
    } else if (ta == 0 && tb == 1) {
      nm_binary_row_scalar_a(op, pa, pb, out, inner);
    } else {
      for (size_t i = 0; i < inner; ++i) {
        out[i] = nm_apply_binary(op, pa[(ptrdiff_t)i * ta],
                                 pb[(ptrdiff_t)i * tb]);
      }
    }
  }
}

/* ---- Benchmark driver --------------------------------------------------- */

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

typedef struct {
  const char *name;
  StBinOp op;
  size_t a_ndim;
  size_t a_dims[ST_MAX_DIMS];
  size_t b_ndim;
  size_t b_dims[ST_MAX_DIMS];
} BenchCase;

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

static void run_case(const BenchCase *bench) {
  FloatTensor *a = make_input(bench->a_ndim, bench->a_dims);
  FloatTensor *b = make_input(bench->b_ndim, bench->b_dims);
  size_t o_ndim = bench->a_ndim > bench->b_ndim ? bench->a_ndim : bench->b_ndim;
  size_t o_dims[ST_MAX_DIMS];
  size_t n = 1;
  for (size_t i = 0; i < o_ndim; ++i) {
    const size_t da =
        i < o_ndim - bench->a_ndim ? 1 : bench->a_dims[i - (o_ndim - bench->a_ndim)];
    const size_t db =
        i < o_ndim - bench->b_ndim ? 1 : bench->b_dims[i - (o_ndim - bench->b_ndim)];
    o_dims[i] = da == 1 ? db : da;
    n *= o_dims[i];
  }
  FloatTensor *out = st_create(o_ndim, o_dims);
  if (out == NULL) {
    exit(1);
  }

  const size_t reps = repetitions_for(n);
  double before_samples[9];
  double after_samples[9];
  for (size_t s = 0; s < 9; ++s) {
    uint64_t start = now_ns();
    for (size_t r = 0; r < reps; ++r) {
      nm_run_binary(bench->op, st_tensor_data(a), bench->a_ndim, bench->a_dims,
                    st_tensor_data(b), bench->b_ndim, bench->b_dims,
                    st_tensor_mutable_data(out), o_ndim, o_dims);
    }
    before_samples[s] = (double)(now_ns() - start) / (double)reps / 1000.0;

    start = now_ns();
    for (size_t r = 0; r < reps; ++r) {
      if (!st_binary_broadcast_into(bench->op, a, b, out)) {
        fprintf(stderr, "kernel failed\n");
        exit(1);
      }
    }
    after_samples[s] = (double)(now_ns() - start) / (double)reps / 1000.0;
  }
  const double before = median(before_samples, 9);
  const double after = median(after_samples, 9);
  printf("| %s | %s | %zu | %.3f | %.3f | %.2fx |\n", bench->name,
         omp_get_max_threads() == 1 ? "1" : "default", n, before, after,
         before / after);

  st_destroy(out);
  st_destroy(b);
  st_destroy(a);
}

/* Threshold sweep: API time at one versus default threads across sizes. */
static void sweep_thread_threshold(void) {
  static const size_t sizes[] = {4096,   16384,   65536,  262144,
                                 1048576, 2097152, 4194304, 8388608};
  const int default_threads = omp_get_max_threads();
  printf("\nThread threshold sweep (st_binary_broadcast_into, µs/call)\n");
  printf("| Elements | Path | 1 thread | default (%d) | parallel gain |\n", default_threads);
  printf("| ---: | --- | ---: | ---: | ---: |\n");
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
    const size_t n = sizes[s];
    const size_t shape[1] = {n};
    FloatTensor *a = make_input(1, shape);
    for (int path = 0; path < 2; ++path) {
      FloatTensor *b;
      const char *label;
      if (path == 0) {
        b = make_input(1, shape);
        label = "same-shape";
      } else {
        b = make_input(1, (size_t[1]){1});
        label = "scalar";
      }
      FloatTensor *out = st_create(1, shape);
      const size_t reps = repetitions_for(n);
      double one_samples[9];
      double many_samples[9];
      omp_set_num_threads(1);
      for (size_t s2 = 0; s2 < 9; ++s2) {
        uint64_t start = now_ns();
        for (size_t r = 0; r < reps; ++r) {
          st_binary_broadcast_into(ST_BIN_ADD, a, b, out);
        }
        one_samples[s2] = (double)(now_ns() - start) / (double)reps / 1000.0;
      }
      omp_set_num_threads(default_threads);
      for (size_t s2 = 0; s2 < 9; ++s2) {
        uint64_t start = now_ns();
        for (size_t r = 0; r < reps; ++r) {
          st_binary_broadcast_into(ST_BIN_ADD, a, b, out);
        }
        many_samples[s2] = (double)(now_ns() - start) / (double)reps / 1000.0;
      }
      const double one = median(one_samples, 9);
      const double many = median(many_samples, 9);
      printf("| %zu | %s | %.3f | %.3f | %.2fx |\n", n, label, one, many,
             one / many);
      st_destroy(out);
      st_destroy(b);
    }
    st_destroy(a);
  }
}

int main(void) {
  printf("| Case | Threads | Elements | nm_lib before (µs) | st_binary (µs) | Speedup |\n");
  printf("| --- | --- | ---: | ---: | ---: | ---: |\n");

  const int default_threads = omp_get_max_threads();
  const BenchCase cases[] = {
      {"same-shape add [8,256,512]", ST_BIN_ADD, 3, {8, 256, 512}, 3, {8, 256, 512}},
      {"same-shape add [1,16,64]", ST_BIN_ADD, 3, {1, 16, 64}, 3, {1, 16, 64}},
      {"bias add [4,128,256]+[256]", ST_BIN_ADD, 3, {4, 128, 256}, 1, {256}},
      {"bias add [4,512,768]+[768]", ST_BIN_ADD, 3, {4, 512, 768}, 1, {768}},
      {"bias add [1,16,64]+[64]", ST_BIN_ADD, 3, {1, 16, 64}, 1, {64}},
      {"attn mask add (walk) [2,4,128,128]+[1,1,128,128]", ST_BIN_ADD, 4, {2, 4, 128, 128}, 4, {1, 1, 128, 128}},
      {"attn mask add (suffix) [2,4,128,128]+[128,128]", ST_BIN_ADD, 4, {2, 4, 128, 128}, 2, {128, 128}},
      {"attn mask add (walk) [1,4,16,16]+[1,1,16,16]", ST_BIN_ADD, 4, {1, 4, 16, 16}, 3, {1, 1, 16, 16}},
      {"scale mul [4,128,256]*[1]", ST_BIN_MUL, 3, {4, 128, 256}, 1, {1}},
      {"scale mul (mid) [4,1,256]*[1,128,1]", ST_BIN_MUL, 3, {4, 1, 256}, 3, {1, 128, 1}},
      {"suffix div [4,128,256]/[256]", ST_BIN_DIV, 3, {4, 128, 256}, 1, {256}},
  };
  const size_t count = sizeof(cases) / sizeof(cases[0]);

  omp_set_num_threads(default_threads);
  for (size_t c = 0; c < count; ++c) {
    run_case(&cases[c]);
  }
  omp_set_num_threads(1);
  for (size_t c = 0; c < count; ++c) {
    run_case(&cases[c]);
  }
  omp_set_num_threads(default_threads);
  sweep_thread_threshold();
  return 0;
}
