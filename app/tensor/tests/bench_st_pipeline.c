/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * bench_st_pipeline — pipeline benchmarks for chained tensor ops.
 *
 * Category: Pipeline
 * Pipelines:
 *   1. Conv2D → BatchNorm2D          (standard CNN block)
 *   2. Conv2D → BatchNorm2D → ReLU   (fused BN+ReLU variant)
 *
 * Output format per case:
 *   [pipeline] <name>  N=.. Cin=.. Cout=.. H=.. W=..
 *     <pipeline>: <ms/iter> ms/iter
 *     counters: mps_hit=<n> mps_miss=<n> ...
 */

#include "st_backend.h"
#include "st_batchnorm.h"
#include "st_conv.h"

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

static void print_counters(void) {
  StBackendCounters c = st_backend_get_counters();
  printf("    counters: mps_hit=%ld  mps_miss=%ld  fallback_gemm=%ld"
         "  fallback_ref=%ld\n",
         c.mps_hit, c.mps_miss, c.fallback_gemm, c.fallback_ref);
}

/* ------------------------------------------------------------------ */
/*  Pipeline case definition                                           */
/* ------------------------------------------------------------------ */

typedef struct PipelineCase {
  const char *name;
  size_t n, c_in, c_out, h, w, k, stride, pad;
  size_t warmup, iters;
} PipelineCase;

/* ------------------------------------------------------------------ */
/*  Pipeline 1: Conv2D → BatchNorm2D                                   */
/* ------------------------------------------------------------------ */

static void bench_conv_bn(const PipelineCase *cfg) {
  StConv2dParams p = st_conv2d_default_params();
  p.stride_h = p.stride_w = cfg->stride;
  p.pad_h    = p.pad_w    = cfg->pad;
  p.backend  = ST_CONV_BACKEND_AUTO;

  size_t out_h = 0, out_w = 0;
  if (!st_conv2d_output_hw(cfg->h, cfg->w, cfg->k, cfg->k, &p, &out_h,
                           &out_w)) {
    fprintf(stderr, "  [SKIP] invalid shape: %s\n", cfg->name);
    return;
  }

  FloatTensor *input   = make4d(cfg->n, cfg->c_in, cfg->h, cfg->w);
  FloatTensor *weight  = make4d(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
  FloatTensor *conv_out = make4d(cfg->n, cfg->c_out, out_h, out_w);
  FloatTensor *bn_out  = make4d(cfg->n, cfg->c_out, out_h, out_w);
  FloatTensor *gamma   = make1d(cfg->c_out);
  FloatTensor *beta    = make1d(cfg->c_out);
  FloatTensor *mean    = make1d(cfg->c_out);
  FloatTensor *var_out = make1d(cfg->c_out);

  if (!input || !weight || !conv_out || !bn_out ||
      !gamma || !beta || !mean || !var_out) {
    fprintf(stderr, "  [OOM] %s\n", cfg->name);
    goto cleanup_conv_bn;
  }

  fill_rand(input,  10u);
  fill_rand(weight, 20u);
  fill_rand(gamma,  30u);
  fill_rand(beta,   40u);

  /* warmup */
  for (size_t i = 0; i < cfg->warmup; ++i) {
    st_conv2d_nchw(input, weight, NULL, &p, conv_out);
    st_batchnorm2d_forward(conv_out, gamma, beta, 1e-5f, bn_out, mean, var_out);
  }

  st_backend_reset_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < cfg->iters; ++i) {
    st_conv2d_nchw(input, weight, NULL, &p, conv_out);
    st_batchnorm2d_forward(conv_out, gamma, beta, 1e-5f, bn_out, mean, var_out);
  }
  uint64_t t1 = now_ns();

  printf("[pipeline] conv→bn  %s\n", cfg->name);
  printf("    N=%zu Cin=%zu Cout=%zu H=%zu W=%zu K=%zu s=%zu p=%zu\n",
         cfg->n, cfg->c_in, cfg->c_out, cfg->h, cfg->w, cfg->k,
         cfg->stride, cfg->pad);
  printf("    conv→bn: %.3f ms/iter\n", elapsed_ms(t0, t1, cfg->iters));
  print_counters();

cleanup_conv_bn:
  st_destroy(input);    st_destroy(weight);
  st_destroy(conv_out); st_destroy(bn_out);
  st_destroy(gamma);    st_destroy(beta);
  st_destroy(mean);     st_destroy(var_out);
}

/* ------------------------------------------------------------------ */
/*  Pipeline 2: Conv2D → BatchNorm2D+ReLU (fused forward)             */
/* ------------------------------------------------------------------ */

static void bench_conv_bn_relu(const PipelineCase *cfg) {
  StConv2dParams p = st_conv2d_default_params();
  p.stride_h = p.stride_w = cfg->stride;
  p.pad_h    = p.pad_w    = cfg->pad;
  p.backend  = ST_CONV_BACKEND_AUTO;

  size_t out_h = 0, out_w = 0;
  if (!st_conv2d_output_hw(cfg->h, cfg->w, cfg->k, cfg->k, &p, &out_h,
                           &out_w)) {
    fprintf(stderr, "  [SKIP] invalid shape: %s\n", cfg->name);
    return;
  }

  FloatTensor *input    = make4d(cfg->n, cfg->c_in, cfg->h, cfg->w);
  FloatTensor *weight   = make4d(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
  FloatTensor *conv_out = make4d(cfg->n, cfg->c_out, out_h, out_w);
  FloatTensor *bn_out   = make4d(cfg->n, cfg->c_out, out_h, out_w);
  FloatTensor *gamma    = make1d(cfg->c_out);
  FloatTensor *beta     = make1d(cfg->c_out);
  FloatTensor *mean     = make1d(cfg->c_out);
  FloatTensor *var_out  = make1d(cfg->c_out);

  if (!input || !weight || !conv_out || !bn_out ||
      !gamma || !beta || !mean || !var_out) {
    fprintf(stderr, "  [OOM] %s\n", cfg->name);
    goto cleanup_conv_bn_relu;
  }

  fill_rand(input,  11u);
  fill_rand(weight, 22u);
  fill_rand(gamma,  33u);
  fill_rand(beta,   44u);

  /* warmup */
  for (size_t i = 0; i < cfg->warmup; ++i) {
    st_conv2d_nchw(input, weight, NULL, &p, conv_out);
    st_batchnorm2d_forward_relu(conv_out, gamma, beta, 1e-5f, bn_out, mean,
                                var_out);
  }

  st_backend_reset_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < cfg->iters; ++i) {
    st_conv2d_nchw(input, weight, NULL, &p, conv_out);
    st_batchnorm2d_forward_relu(conv_out, gamma, beta, 1e-5f, bn_out, mean,
                                var_out);
  }
  uint64_t t1 = now_ns();

  printf("[pipeline] conv→bn+relu  %s\n", cfg->name);
  printf("    N=%zu Cin=%zu Cout=%zu H=%zu W=%zu K=%zu s=%zu p=%zu\n",
         cfg->n, cfg->c_in, cfg->c_out, cfg->h, cfg->w, cfg->k,
         cfg->stride, cfg->pad);
  printf("    conv→bn+relu: %.3f ms/iter\n", elapsed_ms(t0, t1, cfg->iters));
  print_counters();

cleanup_conv_bn_relu:
  st_destroy(input);    st_destroy(weight);
  st_destroy(conv_out); st_destroy(bn_out);
  st_destroy(gamma);    st_destroy(beta);
  st_destroy(mean);     st_destroy(var_out);
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
  printf("=== bench_st_pipeline ===\n\n");

  static const PipelineCase cases[] = {
    /* small: below MPS threshold → CPU */
    { "pipe-small",  1,  8,  8, 16, 16, 3, 1, 1, 2, 20 },
    /* medium */
    { "pipe-medium", 4, 32, 64, 56, 56, 3, 1, 1, 2, 10 },
    /* large: above threshold → MPS on Apple Silicon */
    { "pipe-large",  8, 64,128,112,112, 3, 1, 1, 2,  5 },
  };

  printf("-- Conv → BN --\n");
  for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i)
    bench_conv_bn(&cases[i]);

  printf("\n-- Conv → BN+ReLU --\n");
  for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i)
    bench_conv_bn_relu(&cases[i]);

  printf("\n=== done ===\n");
  return 0;
}
