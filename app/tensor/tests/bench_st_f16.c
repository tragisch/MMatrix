/*
 * Median benchmark for f16 conversion: nm_lib's per-element conversion
 * loop (before) versus the bulk NEON kernels (after).
 * Run: bazel run //app/tensor:bench_st_f16 --config=benchmark
 */

#include "st_f16.h"

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
  float *f32;
  uint16_t *f16;
  size_t n;
} Ctx;

static void f32_to_f16_before(void *context) {
  Ctx *c = context;
  /* nm_lib's loop: one nm_f32_to_f16 call per element. */
  for (size_t i = 0; i < c->n; ++i) {
    c->f16[i] = st_f32_to_f16(c->f32[i]);
  }
}

static void f32_to_f16_after(void *context) {
  Ctx *c = context;
  st_f32_to_f16_bulk(c->f32, c->f16, c->n);
}

static void f16_to_f32_before(void *context) {
  Ctx *c = context;
  for (size_t i = 0; i < c->n; ++i) {
    c->f32[i] = st_f16_to_f32(c->f16[i]);
  }
}

static void f16_to_f32_after(void *context) {
  Ctx *c = context;
  st_f16_to_f32_bulk(c->f16, c->f32, c->n);
}

static void measure(const char *name, size_t n, void (*before)(void *),
                    void (*after)(void *), Ctx *ctx) {
  const size_t reps = repetitions_for(n);
  double before_samples[9];
  double after_samples[9];
  for (size_t s = 0; s < 9; ++s) {
    uint64_t start = now_ns();
    for (size_t r = 0; r < reps; ++r) {
      before(ctx);
    }
    before_samples[s] = (double)(now_ns() - start) / (double)reps / 1000.0;
    start = now_ns();
    for (size_t r = 0; r < reps; ++r) {
      after(ctx);
    }
    after_samples[s] = (double)(now_ns() - start) / (double)reps / 1000.0;
  }
  const double b = median(before_samples, 9);
  const double a = median(after_samples, 9);
  printf("| %s | %zu | %.3f | %.3f | %.2fx |\n", name, n, b, a, b / a);
}

int main(void) {
  printf("| Case | Elements | scalar loop (µs) | bulk (µs) | Speedup |\n");
  printf("| --- | ---: | ---: | ---: | ---: |\n");
  static const size_t sizes[] = {4096, 65536, 262144, 1048576};
  Ctx ctx;
  ctx.n = sizes[sizeof(sizes) / sizeof(sizes[0]) - 1];
  ctx.f32 = malloc(ctx.n * sizeof(float));
  ctx.f16 = malloc(ctx.n * sizeof(uint16_t));
  if (ctx.f32 == NULL || ctx.f16 == NULL) {
    return 1;
  }
  for (size_t i = 0; i < ctx.n; ++i) {
    ctx.f32[i] = sinf((float)i * 0.7351f) * 1.5f + 0.5f;
    ctx.f16[i] = st_f32_to_f16(ctx.f32[i]);
  }
  /* The conversion is single-threaded by design (NEON, no OpenMP). */
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
    ctx.n = sizes[s];
    measure("f32 -> f16", sizes[s], f32_to_f16_before, f32_to_f16_after,
            &ctx);
    measure("f16 -> f32", sizes[s], f16_to_f32_before, f16_to_f32_after, &ctx);
  }
  (void)omp_get_max_threads;
  free(ctx.f16);
  free(ctx.f32);
  return 0;
}
