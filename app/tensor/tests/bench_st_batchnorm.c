/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * bench_st_batchnorm — measures BatchNorm2D GPU-resident vs CPU readback.
 *
 * Two modes are compared for the same BatchNorm2D operation:
 *
 *   metal  : input, gamma, beta, output all allocated with st_create()
 *            (Metal shared buffer on Apple Silicon) → zero-copy GPU residency
 *            → mps_batchnorm2d_forward_preallocated() async encode.
 *
 *   cpu    : input, gamma, beta, output all plain malloc'd pointers → no
 *            Metal handles → MPS fallback to readBytes (Host readback after
 *            GPU compute).
 *
 * The delta (cpu_time − metal_time) is labelled "readback overhead".
 *
 * Category: GPU-resident optimization (Issue 2 validation)
 */

#include "st.h"
#include "st_backend.h"
#include "st_buffer.h"
#include "st_batchnorm.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double elapsed_ms(uint64_t start, uint64_t end, size_t iters) {
  return (double)(end - start) / 1000000.0 / (double)iters;
}

/* Create tensor backed by the best available buffer (Metal on Apple).  */
static FloatTensor *make4d_auto(size_t n, size_t c, size_t h, size_t w) {
  size_t s[4] = {n, c, h, w};
  return st_create(4, s);
}

static FloatTensor *make1d_auto(size_t n) {
  size_t s[1] = {n};
  return st_create(1, s);
}

/* Create tensor backed by a plain malloc pointer (no Metal handle).   */
static FloatTensor *make4d_cpu(size_t n, size_t c, size_t h, size_t w) {
  const size_t numel = n * c * h * w;
  float *data = (float *)calloc(numel, sizeof(float));
  if (!data) return NULL;
  size_t s[4] = {n, c, h, w};
  FloatTensor *t = st_create_with_data(4, s, data, numel, /*take_ownership=*/true);
  return t;
}

static FloatTensor *make1d_cpu(size_t n) {
  float *data = (float *)calloc(n, sizeof(float));
  if (!data) return NULL;
  size_t s[1] = {n};
  FloatTensor *t = st_create_with_data(1, s, data, n, /*take_ownership=*/true);
  return t;
}

static void fill_rand(FloatTensor *t, uint32_t seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1664525u + 1013904223u;
    t->values[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

static void print_counters(void) {
  StBackendCounters c = st_backend_get_counters();
  StBufferMetalAllocatorStats a = st_buffer_metal_allocator_stats_get();
  printf("    counters: mps_hit=%ld  mps_miss=%ld  fallback_gemm=%ld"
         "  fallback_ref=%ld\n",
         c.mps_hit, c.mps_miss, c.fallback_gemm, c.fallback_ref);
  printf("    allocator: requests=%" PRIu64 "  pool_hit=%" PRIu64
         "  new=%" PRIu64 "  stores=%" PRIu64 "  drops=%" PRIu64 "\n",
         a.alloc_requests, a.pool_hits, a.new_allocations,
         a.pool_stores, a.pool_store_drops);
}

/* ------------------------------------------------------------------ */
/*  BatchNorm case                                                     */
/* ------------------------------------------------------------------ */

typedef struct BNormCase {
  const char *name;
  size_t n, c, h, w;
  size_t warmup, iters;
} BNormCase;

static void bench_batchnorm(const BNormCase *cfg) {
  printf("[batchnorm] %s\n", cfg->name);
  printf("    N=%zu C=%zu H=%zu W=%zu  (no inference training outputs)\n",
         cfg->n, cfg->c, cfg->h, cfg->w);

  /* --- Metal (device-resident / zero-copy) --- */
  FloatTensor *in_metal = make4d_auto(cfg->n, cfg->c, cfg->h, cfg->w);
  FloatTensor *gamma_metal = make1d_auto(cfg->c);
  FloatTensor *beta_metal = make1d_auto(cfg->c);
  FloatTensor *out_metal = make4d_auto(cfg->n, cfg->c, cfg->h, cfg->w);
  /* mean/var for inference tracking (GPU-resident) */
  FloatTensor *mean_metal = make1d_auto(cfg->c);
  FloatTensor *var_metal = make1d_auto(cfg->c);

  if (in_metal && gamma_metal && beta_metal && out_metal &&
      mean_metal && var_metal) {
    fill_rand(in_metal, 1u);
    fill_rand(gamma_metal, 2u);
    fill_rand(beta_metal, 3u);

    /* Warmup */
    for (size_t i = 0; i < cfg->warmup; ++i) {
      st_batchnorm2d_forward(in_metal, gamma_metal, beta_metal, 1e-5f,
                             out_metal, mean_metal, var_metal);
    }

    /* Measure Metal path */
    st_backend_reset_counters();
    st_buffer_metal_allocator_stats_reset();
    uint64_t t0 = now_ns();
    for (size_t i = 0; i < cfg->iters; ++i) {
      st_batchnorm2d_forward(in_metal, gamma_metal, beta_metal, 1e-5f,
                             out_metal, mean_metal, var_metal);
    }
    uint64_t t1 = now_ns();
    double ms_metal = elapsed_ms(t0, t1, cfg->iters);
    printf("    metal (GPU-resident): %.3f ms/iter\n", ms_metal);
    print_counters();

    /* --- CPU buffers (with Host readback overhead) --- */
    FloatTensor *in_cpu = make4d_cpu(cfg->n, cfg->c, cfg->h, cfg->w);
    FloatTensor *gamma_cpu = make1d_cpu(cfg->c);
    FloatTensor *beta_cpu = make1d_cpu(cfg->c);
    FloatTensor *out_cpu = make4d_cpu(cfg->n, cfg->c, cfg->h, cfg->w);
    FloatTensor *mean_cpu = make1d_cpu(cfg->c);
    FloatTensor *var_cpu = make1d_cpu(cfg->c);

    if (in_cpu && gamma_cpu && beta_cpu && out_cpu && mean_cpu && var_cpu) {
      fill_rand(in_cpu, 1u);
      fill_rand(gamma_cpu, 2u);
      fill_rand(beta_cpu, 3u);

      /* Warmup */
      for (size_t i = 0; i < cfg->warmup; ++i) {
        st_batchnorm2d_forward(in_cpu, gamma_cpu, beta_cpu, 1e-5f,
                               out_cpu, mean_cpu, var_cpu);
      }

      /* Measure CPU path */
      st_backend_reset_counters();
      st_buffer_metal_allocator_stats_reset();
      uint64_t t2 = now_ns();
      for (size_t i = 0; i < cfg->iters; ++i) {
        st_batchnorm2d_forward(in_cpu, gamma_cpu, beta_cpu, 1e-5f,
                               out_cpu, mean_cpu, var_cpu);
      }
      uint64_t t3 = now_ns();
      double ms_cpu = elapsed_ms(t2, t3, cfg->iters);
      printf("    cpu   (readback):    %.3f ms/iter\n", ms_cpu);
      print_counters();
      printf("    readback overhead:   %+.3f ms/iter  (%.1f%%)\n",
             ms_cpu - ms_metal,
             ms_metal > 0.0 ? (ms_cpu - ms_metal) / ms_metal * 100.0 : 0.0);
    }
    st_destroy(in_cpu);
    st_destroy(gamma_cpu);
    st_destroy(beta_cpu);
    st_destroy(out_cpu);
    st_destroy(mean_cpu);
    st_destroy(var_cpu);
  }
  st_destroy(in_metal);
  st_destroy(gamma_metal);
  st_destroy(beta_metal);
  st_destroy(out_metal);
  st_destroy(mean_metal);
  st_destroy(var_metal);
}

/* ------------------------------------------------------------------ */
/*  Entry point                                                        */
/* ------------------------------------------------------------------ */

int main(void) {
  printf("=== bench_st_batchnorm ===\n");
  printf("    metal  =GPU-resident (initWithMTLBuffer:, async encode)\n");
  printf("    cpu    =readback (Host readBytes after GPU compute)\n\n");

  BNormCase cases[] = {
      /* small: inference on embedding layer */
      {"bn-small", 16, 64, 28, 28, 10, 50},
      /* medium: inference on typical conv layer */
      {"bn-medium", 8, 256, 56, 56, 10, 50},
      /* large: inference on high-res layer */
      {"bn-large", 4, 512, 112, 112, 5, 20},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    bench_batchnorm(&cases[i]);
    printf("\n");
  }

  printf("=== done ===\n");
  return 0;
}
