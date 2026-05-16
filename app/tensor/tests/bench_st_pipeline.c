/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * bench_st_pipeline — fused pipeline benchmarks for chained tensor ops.
 *
 * Goal:
 *   Measure GPU-resident pipeline behavior with sync only at boundary,
 *   and compare against per-iteration sync.
 *
 * Pipelines:
 *   1) fused Conv2D + BatchNorm2D
 *   2) fused Conv2D + BatchNorm2D + Pool2D
 */

#include "st_backend.h"
#include "st_conv.h"
#include "st_pool.h"
#include "log.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

static FloatTensor *make4d(size_t n, size_t c, size_t h, size_t w) {
  size_t s[4] = {n, c, h, w};
  return st_create(4, s);
}

static FloatTensor *make1d(size_t n) {
  size_t s[1] = {n};
  return st_create(1, s);
}

static void fill_rand(FloatTensor *t, uint32_t seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1664525u + 1013904223u;
    t->values[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

static const char *buf_type(const FloatTensor *t) {
  if (t && t->buf && st_buffer_metal_handle(t->buf) != NULL) {
    return "metal";
  }
  return "cpu";
}

static void print_counters(void) {
  StBackendCounters c = st_backend_get_counters();
  printf("    counters: mps_hit=%ld  mps_miss=%ld  fallback_gemm=%ld"
         "  fallback_ref=%ld  readbytes=%ld  fastpath=%ld\n",
         c.mps_hit, c.mps_miss, c.fallback_gemm, c.fallback_ref,
         c.conv_readbytes, c.conv_fastpath_hit);
}

/* ------------------------------------------------------------------ */
/*  Pipeline case definition                                           */
/* ------------------------------------------------------------------ */

typedef struct PipelineCase {
  const char *name;
  size_t n, c_in, c_out, h, w, k, stride, pad;
  size_t warmup, iters;
} PipelineCase;

static void print_case_header(const char *pipeline,
                              const char *variant,
                              const PipelineCase *cfg,
                              const FloatTensor *out) {
  printf("[pipeline] %s  %s  %s\n", pipeline, variant, cfg->name);
  printf("    N=%zu Cin=%zu Cout=%zu H=%zu W=%zu K=%zu s=%zu p=%zu  out_buf=%s\n",
         cfg->n, cfg->c_in, cfg->c_out, cfg->h, cfg->w, cfg->k,
         cfg->stride, cfg->pad, buf_type(out));
}

/* ------------------------------------------------------------------ */
/*  Pipeline 1: fused Conv2D + BatchNorm2D                             */
/* ------------------------------------------------------------------ */

static void bench_fused_conv_bn(const PipelineCase *cfg,
                                bool sync_each_iter) {
  StConv2dParams p = st_conv2d_default_params();
  p.stride_h = p.stride_w = cfg->stride;
  p.pad_h    = p.pad_w    = cfg->pad;
  p.backend  = ST_CONV_BACKEND_MPS;

  size_t out_h = 0, out_w = 0;
  if (!st_conv2d_output_hw(cfg->h, cfg->w, cfg->k, cfg->k, &p, &out_h,
                           &out_w)) {
    fprintf(stderr, "  [SKIP] invalid shape: %s\n", cfg->name);
    return;
  }

  FloatTensor *input  = make4d(cfg->n, cfg->c_in, cfg->h, cfg->w);
  FloatTensor *weight = make4d(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
  FloatTensor *out    = make4d(cfg->n, cfg->c_out, out_h, out_w);
  FloatTensor *gamma  = make1d(cfg->c_out);
  FloatTensor *beta   = make1d(cfg->c_out);

  if (!input || !weight || !out || !gamma || !beta) {
    fprintf(stderr, "  [OOM] %s\n", cfg->name);
    goto cleanup_conv_bn;
  }

  fill_rand(input,  10u);
  fill_rand(weight, 20u);
  fill_rand(gamma,  30u);
  fill_rand(beta,   40u);

  /* warmup */
  for (size_t i = 0; i < cfg->warmup; ++i) {
    if (!st_conv2d_batchnorm2d_forward_nchw(
            input, weight, NULL, &p,
            gamma, beta, 1e-5f,
            out, NULL, NULL,
            false)) {
      fprintf(stderr, "  [SKIP] fused conv+bn unavailable: %s\n", cfg->name);
      goto cleanup_conv_bn;
    }
    if (sync_each_iter) {
      st_tensor_sync(out);
    }
  }

  st_backend_reset_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < cfg->iters; ++i) {
    if (!st_conv2d_batchnorm2d_forward_nchw(
            input, weight, NULL, &p,
            gamma, beta, 1e-5f,
            out, NULL, NULL,
            false)) {
      fprintf(stderr, "  [SKIP] fused conv+bn unavailable: %s\n", cfg->name);
      goto cleanup_conv_bn;
    }
    if (sync_each_iter) {
      st_tensor_sync(out);
    }
  }
  if (!sync_each_iter) {
    st_tensor_sync(out);
  }
  uint64_t t1 = now_ns();

  print_case_header("conv+bn(fused)",
                    sync_each_iter ? "sync_each_iter" : "boundary_sync_only",
                    cfg, out);
  printf("    time: %.3f ms/iter\n", elapsed_ms(t0, t1, cfg->iters));
  print_counters();

cleanup_conv_bn:
  st_destroy(input);
  st_destroy(weight);
  st_destroy(out);
  st_destroy(gamma);
  st_destroy(beta);
}

/* ------------------------------------------------------------------ */
/*  Pipeline 2: fused Conv2D + BatchNorm2D + Pool2D                    */
/* ------------------------------------------------------------------ */

static void bench_fused_conv_bn_pool(const PipelineCase *cfg,
                                     bool sync_each_iter) {
  StConv2dParams p = st_conv2d_default_params();
  p.stride_h = p.stride_w = cfg->stride;
  p.pad_h    = p.pad_w    = cfg->pad;
  p.backend  = ST_CONV_BACKEND_MPS;

  StPool2dParams pool;
  pool.pool_type = ST_POOL_MAX;
  pool.kernel_h = 2;
  pool.kernel_w = 2;
  pool.stride_h = 2;
  pool.stride_w = 2;
  pool.pad_h = 0;
  pool.pad_w = 0;

  size_t out_h = 0, out_w = 0;
  if (!st_conv2d_output_hw(cfg->h, cfg->w, cfg->k, cfg->k, &p, &out_h,
                           &out_w)) {
    fprintf(stderr, "  [SKIP] invalid shape: %s\n", cfg->name);
    return;
  }

  size_t pool_out_h = 0, pool_out_w = 0;
  if (!st_pool2d_output_hw(out_h, out_w,
                           pool.kernel_h, pool.kernel_w,
                           pool.stride_h, pool.stride_w,
                           pool.pad_h, pool.pad_w,
                           &pool_out_h, &pool_out_w)) {
    fprintf(stderr, "  [SKIP] invalid pool shape: %s\n", cfg->name);
    return;
  }

  FloatTensor *input  = make4d(cfg->n, cfg->c_in, cfg->h, cfg->w);
  FloatTensor *weight = make4d(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
  FloatTensor *out    = make4d(cfg->n, cfg->c_out, pool_out_h, pool_out_w);
  FloatTensor *gamma  = make1d(cfg->c_out);
  FloatTensor *beta   = make1d(cfg->c_out);

  if (!input || !weight || !out || !gamma || !beta) {
    fprintf(stderr, "  [OOM] %s\n", cfg->name);
    goto cleanup_conv_bn_relu;
  }

  fill_rand(input,  11u);
  fill_rand(weight, 22u);
  fill_rand(gamma,  33u);
  fill_rand(beta,   44u);

  /* warmup */
  for (size_t i = 0; i < cfg->warmup; ++i) {
    if (!st_conv2d_batchnorm2d_pool_forward_nchw(
            input, weight, NULL, &p,
            gamma, beta, 1e-5f,
            &pool,
            out, NULL, NULL,
            false)) {
      fprintf(stderr, "  [SKIP] fused conv+bn+pool unavailable: %s\n", cfg->name);
      goto cleanup_conv_bn_relu;
    }
    if (sync_each_iter) {
      st_tensor_sync(out);
    }
  }

  st_backend_reset_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < cfg->iters; ++i) {
    if (!st_conv2d_batchnorm2d_pool_forward_nchw(
            input, weight, NULL, &p,
            gamma, beta, 1e-5f,
            &pool,
            out, NULL, NULL,
            false)) {
      fprintf(stderr, "  [SKIP] fused conv+bn+pool unavailable: %s\n", cfg->name);
      goto cleanup_conv_bn_relu;
    }
    if (sync_each_iter) {
      st_tensor_sync(out);
    }
  }
  if (!sync_each_iter) {
    st_tensor_sync(out);
  }
  uint64_t t1 = now_ns();

  print_case_header("conv+bn+pool(fused)",
                    sync_each_iter ? "sync_each_iter" : "boundary_sync_only",
                    cfg, out);
  printf("    time: %.3f ms/iter\n", elapsed_ms(t0, t1, cfg->iters));
  print_counters();

cleanup_conv_bn_relu:
  st_destroy(input);
  st_destroy(weight);
  st_destroy(out);
  st_destroy(gamma);
  st_destroy(beta);
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
  const bool prev_async = st_backend_get_conv_mps_async();
  st_backend_set_conv_mps_async(true);
  log_set_level(LOG_WARN);
  printf("=== bench_st_pipeline ===\n\n");

  static const PipelineCase cases[] = {
    /* small: forced-MPS stress case; may skip if MPS rejects the shape */
    { "pipe-small",  1,  8,  8, 16, 16, 3, 1, 1, 2, 20 },
    /* medium */
    { "pipe-medium", 4, 32, 64, 56, 56, 3, 1, 1, 2, 10 },
    /* large */
    { "pipe-large",  8, 64,128,112,112, 3, 1, 1, 2,  5 },
  };

  printf("-- Fused Conv+BN --\n");
  for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i)
    bench_fused_conv_bn(&cases[i], false); /* boundary sync only */

  printf("\n-- Fused Conv+BN (sync each iter) --\n");
  for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i)
    bench_fused_conv_bn(&cases[i], true);

  printf("\n-- Fused Conv+BN+Pool --\n");
  for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i)
    bench_fused_conv_bn_pool(&cases[i], false); /* boundary sync only */

  printf("\n-- Fused Conv+BN+Pool (sync each iter) --\n");
  for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i)
    bench_fused_conv_bn_pool(&cases[i], true);

  printf("\n=== done ===\n");
  st_backend_set_conv_mps_async(prev_async);
  return 0;
}
