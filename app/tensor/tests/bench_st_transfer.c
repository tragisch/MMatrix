/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * bench_st_transfer — measures Host↔Device transfer overhead.
 *
 * Two modes are compared for the same Conv2D + BatchNorm2D pipeline:
 *
 *   metal  : tensors allocated with st_create() → auto-selects Metal shared
 *            buffer on Apple Silicon → zero-copy (initWithMTLBuffer:).
 *
 *   cpu    : tensors allocated with st_create_with_data() over a plain
 *            malloc'd pointer → no Metal handle → MPS wraps via NSData
 *            (initWithDevice:data:…) which creates a temporary GPU blit.
 *
 * The delta  (cpu_time − metal_time)  is labelled "transfer overhead".
 *
 * On non-Apple platforms the MPS backend is absent; both modes fall
 * through to the CPU GEMM/reference path and the delta reflects only
 * allocation / cache differences.
 *
 * Category: E2E vs device-resident
 */

#include "st.h"
#include "st_backend.h"
#include "st_buffer.h"
#include "st_batchnorm.h"
#include "st_conv.h"

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
/*  Transfer case                                                      */
/* ------------------------------------------------------------------ */

typedef struct TransferCase {
  const char *name;
  size_t n, c_in, c_out, h, w, k, stride, pad;
  size_t warmup, iters;
} TransferCase;

static void bench_transfer(const TransferCase *cfg) {
  StConv2dParams p = st_conv2d_default_params();
  p.stride_h = p.stride_w = cfg->stride;
  p.pad_h    = p.pad_w    = cfg->pad;
  p.backend  = ST_CONV_BACKEND_AUTO;

  size_t out_h = 0, out_w = 0;
  if (!st_conv2d_output_hw(cfg->h, cfg->w, cfg->k, cfg->k, &p, &out_h, &out_w)) {
    fprintf(stderr, "  [SKIP] invalid shape: %s\n", cfg->name);
    return;
  }

  printf("[transfer] %s\n", cfg->name);
  printf("    N=%zu Cin=%zu Cout=%zu H=%zu W=%zu K=%zu s=%zu p=%zu\n",
         cfg->n, cfg->c_in, cfg->c_out, cfg->h, cfg->w, cfg->k,
         cfg->stride, cfg->pad);

  /* --- Metal (device-resident / zero-copy) --- */
  FloatTensor *i1 = make4d_auto(cfg->n, cfg->c_in, cfg->h, cfg->w);
  FloatTensor *w1 = make4d_auto(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
  FloatTensor *co1= make4d_auto(cfg->n, cfg->c_out, out_h, out_w);
  FloatTensor *bo1= make4d_auto(cfg->n, cfg->c_out, out_h, out_w);
  FloatTensor *g1 = make1d_auto(cfg->c_out);
  FloatTensor *b1 = make1d_auto(cfg->c_out);
  FloatTensor *m1 = make1d_auto(cfg->c_out);
  FloatTensor *v1 = make1d_auto(cfg->c_out);
  if (i1 && w1 && co1 && bo1 && g1 && b1 && m1 && v1) {
    fill_rand(i1, 1u); fill_rand(w1, 2u);
    fill_rand(g1, 3u); fill_rand(b1, 4u);
    for (size_t i = 0; i < cfg->warmup; ++i) {
      st_conv2d_nchw(i1, w1, NULL, &p, co1);
      st_batchnorm2d_forward(co1, g1, b1, 1e-5f, bo1, m1, v1);
    }
    st_backend_reset_counters();
    uint64_t t0 = now_ns();
    for (size_t i = 0; i < cfg->iters; ++i) {
      st_conv2d_nchw(i1, w1, NULL, &p, co1);
      st_batchnorm2d_forward(co1, g1, b1, 1e-5f, bo1, m1, v1);
    }
    uint64_t t1 = now_ns();
    double ms_metal = elapsed_ms(t0, t1, cfg->iters);
    printf("    metal (zero-copy):  %.3f ms/iter\n", ms_metal);
    print_counters();

    /* --- CPU buffers (with Host→Device blit overhead) --- */
    FloatTensor *i2 = make4d_cpu(cfg->n, cfg->c_in, cfg->h, cfg->w);
    FloatTensor *w2 = make4d_cpu(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
    FloatTensor *co2= make4d_cpu(cfg->n, cfg->c_out, out_h, out_w);
    FloatTensor *bo2= make4d_cpu(cfg->n, cfg->c_out, out_h, out_w);
    FloatTensor *g2 = make1d_cpu(cfg->c_out);
    FloatTensor *b2 = make1d_cpu(cfg->c_out);
    FloatTensor *m2 = make1d_cpu(cfg->c_out);
    FloatTensor *v2 = make1d_cpu(cfg->c_out);
    if (i2 && w2 && co2 && bo2 && g2 && b2 && m2 && v2) {
      fill_rand(i2, 1u); fill_rand(w2, 2u);
      fill_rand(g2, 3u); fill_rand(b2, 4u);
      for (size_t i = 0; i < cfg->warmup; ++i) {
        st_conv2d_nchw(i2, w2, NULL, &p, co2);
        st_batchnorm2d_forward(co2, g2, b2, 1e-5f, bo2, m2, v2);
      }
      st_backend_reset_counters();
      uint64_t t2 = now_ns();
      for (size_t i = 0; i < cfg->iters; ++i) {
        st_conv2d_nchw(i2, w2, NULL, &p, co2);
        st_batchnorm2d_forward(co2, g2, b2, 1e-5f, bo2, m2, v2);
      }
      uint64_t t3 = now_ns();
      double ms_cpu = elapsed_ms(t2, t3, cfg->iters);
      printf("    cpu   (blit copy):  %.3f ms/iter\n", ms_cpu);
      print_counters();
      printf("    transfer overhead:  %+.3f ms/iter  (%.1f%%)\n",
             ms_cpu - ms_metal,
             ms_metal > 0.0 ? (ms_cpu - ms_metal) / ms_metal * 100.0 : 0.0);
    }
    st_destroy(i2);  st_destroy(w2);  st_destroy(co2); st_destroy(bo2);
    st_destroy(g2);  st_destroy(b2);  st_destroy(m2);  st_destroy(v2);
  }
  st_destroy(i1);  st_destroy(w1);  st_destroy(co1); st_destroy(bo1);
  st_destroy(g1);  st_destroy(b1);  st_destroy(m1);  st_destroy(v1);
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
  printf("=== bench_st_transfer ===\n");
  printf("    metal=zero-copy (initWithMTLBuffer:)\n");
  printf("    cpu  =blit copy (initWithDevice:data:…)\n\n");

  static const TransferCase cases[] = {
    /* large: above MPS threshold → MPS selected, transfer effect visible */
    { "transfer-medium", 4, 32,  64,  56,  56, 3, 1, 1, 3, 10 },
    { "transfer-large",  8, 64, 128, 112, 112, 3, 1, 1, 3,  5 },
  };

  for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
    bench_transfer(&cases[i]);
    printf("\n");
  }

  printf("=== done ===\n");
  return 0;
}
