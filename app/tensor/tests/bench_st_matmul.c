/*
 * Median benchmark for batched matmul: nm_lib's run_matmul loop (before)
 * versus st_matmul_batched (after).
 * Run: bazel run //app/tensor:bench_st_matmul --config=benchmark
 */

#include "sm.h"
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
  if (n <= 4194304) return 40;
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

/* nm_lib's run_matmul: sequential sm_gemm per batch, with the single-matrix
 * stacking fast path. */
static bool nm_gemm(const float *a, const float *b, float *c, size_t m,
                    size_t k, size_t n) {
  FloatMatrix ma = {.rows = m, .cols = k, .capacity = m * k, .values = (float *)a};
  FloatMatrix mb = {.rows = k, .cols = n, .capacity = k * n, .values = (float *)b};
  FloatMatrix mc = {.rows = m, .cols = n, .capacity = m * n, .values = c};
  return sm_gemm(&mc, 1.0f, &ma, SM_NO_TRANSPOSE, &mb, SM_NO_TRANSPOSE, 0.0f);
}

static void nm_run_matmul(const float *a, size_t a_ndim, const size_t *a_dims,
                          const float *b, size_t b_ndim, const size_t *b_dims,
                          float *out) {
  const size_t m = a_dims[a_ndim - 2];
  const size_t k = a_dims[a_ndim - 1];
  const size_t n = b_dims[b_ndim - 1];
  size_t b_numel = 1;
  for (size_t i = 0; i < b_ndim; ++i) {
    b_numel *= b_dims[i];
  }
  if (b_numel == k * n) {
    size_t a_numel = 1;
    for (size_t i = 0; i < a_ndim; ++i) {
      a_numel *= a_dims[i];
    }
    nm_gemm(a, b, out, a_numel / k, k, n);
    return;
  }
  const size_t batch_a = a_ndim - 2;
  const size_t batch_b = b_ndim - 2;
  const size_t batch_rank = batch_a > batch_b ? batch_a : batch_b;
  size_t da[BENCH_DIMS] = {0};
  size_t db[BENCH_DIMS] = {0};
  size_t d[BENCH_DIMS] = {0};
  for (size_t i = 0; i < batch_rank; ++i) {
    da[i] = i < batch_rank - batch_a ? 1 : a_dims[i - (batch_rank - batch_a)];
    db[i] = i < batch_rank - batch_b ? 1 : b_dims[i - (batch_rank - batch_b)];
    d[i] = da[i] == 1 ? db[i] : da[i];
  }
  size_t batches = 1;
  for (size_t i = 0; i < batch_rank; ++i) {
    batches *= d[i];
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
    nm_gemm(a + offset_a * m * k, b + offset_b * k * n, out + index * m * n, m,
            k, n);
  }
}

typedef struct {
  const char *name;
  size_t a_ndim;
  size_t a_dims[ST_MAX_DIMS];
  size_t b_ndim;
  size_t b_dims[ST_MAX_DIMS];
} MatmulCase;

static void run_case(const MatmulCase *bench) {
  FloatTensor *a = make_input(bench->a_ndim, bench->a_dims);
  FloatTensor *b = make_input(bench->b_ndim, bench->b_dims);
  const size_t m = bench->a_dims[bench->a_ndim - 2];
  const size_t n = bench->b_dims[bench->b_ndim - 1];
  const size_t batch_a = bench->a_ndim - 2;
  const size_t batch_b = bench->b_ndim - 2;
  const size_t batch_rank = batch_a > batch_b ? batch_a : batch_b;
  size_t out_dims[ST_MAX_DIMS];
  size_t out_ndim = 0;
  size_t batches = 1;
  for (size_t i = 0; i < batch_rank; ++i) {
    const size_t da =
        i < batch_rank - batch_a ? 1 : bench->a_dims[i - (batch_rank - batch_a)];
    const size_t db =
        i < batch_rank - batch_b ? 1 : bench->b_dims[i - (batch_rank - batch_b)];
    const size_t d = da == 1 ? db : da;
    out_dims[out_ndim++] = d;
    batches *= d;
  }
  out_dims[out_ndim++] = m;
  out_dims[out_ndim++] = n;
  FloatTensor *c = st_create(out_ndim, out_dims);
  if (c == NULL) {
    exit(1);
  }

  const size_t work = st_tensor_numel(a) + st_tensor_numel(b) +
                      st_tensor_numel(c);
  const size_t reps = repetitions_for(work);
  double before_samples[9];
  double after_samples[9];
  for (size_t s = 0; s < 9; ++s) {
    uint64_t start = now_ns();
    for (size_t r = 0; r < reps; ++r) {
      nm_run_matmul(st_tensor_data(a), bench->a_ndim, bench->a_dims,
                    st_tensor_data(b), bench->b_ndim, bench->b_dims,
                    st_tensor_mutable_data(c));
    }
    before_samples[s] = (double)(now_ns() - start) / (double)reps / 1000.0;

    start = now_ns();
    for (size_t r = 0; r < reps; ++r) {
      if (!st_matmul_batched(a, b, c)) {
        fprintf(stderr, "st_matmul_batched failed\n");
        exit(1);
      }
    }
    after_samples[s] = (double)(now_ns() - start) / (double)reps / 1000.0;
  }
  const double before = median(before_samples, 9);
  const double after = median(after_samples, 9);
  printf("| %s | %s | %zu | %zu | %.3f | %.3f | %.2fx |\n", bench->name,
         omp_get_max_threads() == 1 ? "1" : "default", batches, m * m * n,
         before, after, before / after);

  st_destroy(c);
  st_destroy(b);
  st_destroy(a);
}

int main(void) {
  printf("| Case | Threads | Batches | GEMM MxKxN | nm_lib before (µs) | st_matmul_batched (µs) | Speedup |\n");
  printf("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n");
  const int default_threads = omp_get_max_threads();

  const MatmulCase cases[] = {
      {"proj stacked [4,128,256]x[256,512]", 3, {4, 128, 256}, 2, {256, 512}},
      {"proj stacked [8,256,512]x[512,512]", 3, {8, 256, 512}, 2, {512, 512}},
      {"proj stacked [1,16,64]x[64,64]", 3, {1, 16, 64}, 2, {64, 64}},
      {"batched [2,4,128,64]x[2,4,64,128] (attn scores)", 4, {2, 4, 128, 64}, 4, {2, 4, 64, 128}},
      {"batched [1,4,16,64]x[1,4,64,16] (small, many)", 4, {1, 4, 16, 64}, 4, {1, 4, 64, 16}},
      {"batched [2,4,128,64]x[2,1,64,128] (shared head)", 4, {2, 4, 128, 64}, 4, {2, 1, 64, 128}},
      {"batched [4,12,64,64]x[4,12,64,64]", 4, {4, 12, 64, 64}, 4, {4, 12, 64, 64}},
  };
  const size_t count = sizeof(cases) / sizeof(cases[0]);

  omp_set_num_threads(default_threads);
  for (size_t i = 0; i < count; ++i) {
    run_case(&cases[i]);
  }
  omp_set_num_threads(1);
  for (size_t i = 0; i < count; ++i) {
    run_case(&cases[i]);
  }
  return 0;
}
