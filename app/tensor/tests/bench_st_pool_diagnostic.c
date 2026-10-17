/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * bench_st_pool_diagnostic — isolates MaxPool2D vs AvgPool2D overhead.
 *
 * Investigates: Issue 1 xlarge-anomalie (168ms MaxPool vs 45ms AvgPool)
 *
 * Hypothesis:
 * - MaxPool kernel more complex (element-wise max + implicit arg-max)?
 * - GPU queue/sync overhead visible at larger tensor scales?
 * - MPSGraph compilation or execution efficiency difference?
 *
 * Test matrix:
 * - Kernel sizes: 2x2, 3x3, 5x5 (variable complexity)
 * - Tensor sizes: small, medium, large, xlarge (sync overhead scaling)
 * - Metal GPU-resident path only (no readBytes fallback)
 */

#include "st.h"
#include "st_backend.h"
#include "st_buffer.h"
#include "st_pool.h"

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

/* Create tensor backed by Metal (best available).  */
static FloatTensor *make4d_metal(size_t n, size_t c, size_t h, size_t w) {
  size_t s[4] = {n, c, h, w};
  return st_create(4, s);
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
  printf("    mps_hit=%ld  fallback=%ld | "
         "alloc_req=%" PRIu64 "  pool_hit=%" PRIu64 "\n",
         c.mps_hit, c.fallback_gemm + c.fallback_ref,
         a.alloc_requests, a.pool_hits);
}

/* ------------------------------------------------------------------ */
/*  Pool test case                                                     */
/* ------------------------------------------------------------------ */

typedef struct PoolCase {
  const char *name;
  size_t n, c, h, w;  /* input shape */
  size_t k, stride;   /* kernel size, stride */
  size_t iters, warmup;
} PoolCase;

typedef struct PoolResults {
  double maxpool_ms;
  double avgpool_ms;
  double delta_ms;
  double pct_diff;
} PoolResults;

static PoolResults bench_maxpool_vs_avgpool(const PoolCase *cfg) {
  PoolResults res = {0};
  
  /* Pad for stride (simple: pad_h = pad_w = k/2) */
  size_t pad = cfg->k / 2;
  size_t oh = (cfg->h + 2*pad - cfg->k) / cfg->stride + 1;
  size_t ow = (cfg->w + 2*pad - cfg->k) / cfg->stride + 1;

  printf("[pool] %s (input %zu×%zu×%zu×%zu, kernel %zu, stride %zu)\n",
         cfg->name, cfg->n, cfg->c, cfg->h, cfg->w, cfg->k, cfg->stride);

  /* === MaxPool2D === */
  FloatTensor *in_max = make4d_metal(cfg->n, cfg->c, cfg->h, cfg->w);
  FloatTensor *out_max = make4d_metal(cfg->n, cfg->c, oh, ow);

  if (in_max && out_max) {
    fill_rand(in_max, 123u);

    /* Warmup */
    for (size_t i = 0; i < cfg->warmup; ++i) {
      st_maxpool2d_nchw(in_max, cfg->k, cfg->k,
                        cfg->stride, cfg->stride, pad, pad, out_max, NULL);
    }

    st_backend_reset_counters();
    st_buffer_metal_allocator_stats_reset();
    uint64_t t0 = now_ns();
    for (size_t i = 0; i < cfg->iters; ++i) {
      st_maxpool2d_nchw(in_max, cfg->k, cfg->k,
                        cfg->stride, cfg->stride, pad, pad, out_max, NULL);
    }
    uint64_t t1 = now_ns();
    res.maxpool_ms = elapsed_ms(t0, t1, cfg->iters);
    printf("  MaxPool2D:  %.3f ms/iter | ", res.maxpool_ms);
    print_counters();
  }
  st_destroy(in_max);
  st_destroy(out_max);

  /* === AvgPool2D === */
  FloatTensor *in_avg = make4d_metal(cfg->n, cfg->c, cfg->h, cfg->w);
  FloatTensor *out_avg = make4d_metal(cfg->n, cfg->c, oh, ow);

  if (in_avg && out_avg) {
    fill_rand(in_avg, 456u);

    /* Warmup */
    for (size_t i = 0; i < cfg->warmup; ++i) {
      st_avgpool2d_nchw(in_avg, cfg->k, cfg->k,
                        cfg->stride, cfg->stride, pad, pad, out_avg);
    }

    st_backend_reset_counters();
    st_buffer_metal_allocator_stats_reset();
    uint64_t t2 = now_ns();
    for (size_t i = 0; i < cfg->iters; ++i) {
      st_avgpool2d_nchw(in_avg, cfg->k, cfg->k,
                        cfg->stride, cfg->stride, pad, pad, out_avg);
    }
    uint64_t t3 = now_ns();
    res.avgpool_ms = elapsed_ms(t2, t3, cfg->iters);
    printf("  AvgPool2D:  %.3f ms/iter | ", res.avgpool_ms);
    print_counters();
  }
  st_destroy(in_avg);
  st_destroy(out_avg);

  res.delta_ms = res.maxpool_ms - res.avgpool_ms;
  res.pct_diff = (res.avgpool_ms > 0.0) ? 
      (res.maxpool_ms - res.avgpool_ms) / res.avgpool_ms * 100.0 : 0.0;
  printf("  Δ = %+.3f ms  (MaxPool %+.1f%% vs AvgPool)\n\n",
         res.delta_ms, res.pct_diff);

  return res;
}

/* ================================================================== */
/*  Entry point                                                        */
/* ================================================================== */

int main(void) {
  printf("=== bench_st_pool_diagnostic ===\n");
  printf("Issue 1 xlarge-anomalie investigation: MaxPool vs AvgPool overhead\n");
  printf("All tensors Metal-resident (GPU fastpath, no readBytes fallback)\n\n");

  PoolCase cases[] = {
      /* Test scaling: kernel size fixed (3x3), vary tensor size */
      {"small-k3",      16, 64,  28,  28, 3, 1, 50, 10},
      {"medium-k3",      8, 128, 56,  56, 3, 1, 50, 10},
      {"large-k3",       4, 256, 112, 112, 3, 1, 30, 5},
      {"xlarge-k3",      2, 512, 224, 224, 3, 1, 10, 3},

      /* Test kernel complexity: tensor fixed (large), vary kernel size */
      {"large-k2",       4, 256, 112, 112, 2, 1, 30, 5},
      {"large-k3",       4, 256, 112, 112, 3, 1, 30, 5},
      {"large-k5",       4, 256, 112, 112, 5, 1, 30, 5},
  };

  printf("=== SCALING TEST (kernel=3×3, varying tensor size) ===\n");
  for (size_t i = 0; i < 4; ++i) {
    bench_maxpool_vs_avgpool(&cases[i]);
  }

  printf("\n=== KERNEL COMPLEXITY TEST (tensor=large, varying kernel size) ===\n");
  for (size_t i = 4; i < 7; ++i) {
    bench_maxpool_vs_avgpool(&cases[i]);
  }

  printf("=== done ===\n");
  return 0;
}
