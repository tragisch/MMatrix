/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * bench_st_single_op — isolated single-op benchmarks for tensor ops.
 *
 * Category: Single-Op
 * Ops covered: Conv2D (AUTO), MaxPool2D, AvgPool2D, BatchNorm2D
 *
 * Output format per case:
 *   <name>  N=.. Cin=.. Cout=.. H=.. W=..
 *     <op>: <ms/iter> ms/iter   mps_hit=<n> mps_miss=<n>
 */

#include "st_backend.h"
#include "st_batchnorm.h"
#include "st_conv.h"
#include "st_pool.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
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

static int bench_csv_enabled(void) {
  const char *env = getenv("BENCH_CSV");
  return env && strcmp(env, "0") != 0;
}

static const char *bench_async_profile(void) {
  const char *p = getenv("MMATRIX_ST_ASYNC_PROFILE");
  return (p && p[0] != '\0') ? p : "default";
}

static void print_csv_header(void) {
  printf("suite,async_profile,op,case,N,Cin,Cout,H,W,K,stride,pad,ms_per_iter,mps_hit,mps_miss,fallback_gemm,fallback_ref,max_abs,max_rel\n");
}

static void print_csv_row(const char *op, const char *name,
                          size_t n, size_t cin, size_t cout,
                          size_t h, size_t w, size_t k,
                          size_t stride, size_t pad,
                          double ms, StBackendCounters c,
                          float max_abs, float max_rel) {
  printf("single_op,%s,%s,%s,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%.6f,%ld,%ld,%ld,%ld,%.6g,%.6g\n",
    bench_async_profile(), op, name, n, cin, cout, h, w, k, stride, pad, ms,
         c.mps_hit, c.mps_miss, c.fallback_gemm, c.fallback_ref,
         max_abs, max_rel);
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

/* ------------------------------------------------------------------ */
/*  Accuracy helpers                                                   */
/* ------------------------------------------------------------------ */

/*
 * Compute element-wise max_abs and max_rel error between `out` and `ref`.
 * Relative error uses max(|out|, |ref|) as denominator (robust near zero).
 * F32 tolerance: max_abs < 1e-3, max_rel < 0.1 (10 %).
 */
static void accuracy(const FloatTensor *out, const FloatTensor *ref,
                     float *max_abs_out, float *max_rel_out) {
  float max_abs = 0.0f, max_rel = 0.0f;
  for (size_t i = 0; i < out->numel; ++i) {
    float a = fabsf(out->values[i] - ref->values[i]);
    float denom = fmaxf(fabsf(out->values[i]), fabsf(ref->values[i]));
    float r = denom > 1e-6f ? a / denom : a;
    if (a > max_abs) max_abs = a;
    if (r > max_rel) max_rel = r;
  }
  *max_abs_out = max_abs;
  *max_rel_out = max_rel;
}

static void print_accuracy(float max_abs, float max_rel) {
  /*
   * F32 tolerance (WARN when exceeded):
   *   max_abs > 1e-3  (absolute error — primary signal)
   *   max_rel > 0.10 AND max_abs > 1e-5  (relative — only meaningful above noise floor)
   */
  const char *flag =
      (max_abs > 1e-3f || (max_rel > 0.10f && max_abs > 5e-5f)) ? "  [WARN]" : "";
  printf("    accuracy(vs ref):  max_abs=%.2e  max_rel=%.2e%s\n",
         (double)max_abs, (double)max_rel, flag);
}

/* Inline reference: MaxPool2D (NCHW, no indices). */
static void ref_maxpool2d(const FloatTensor *in,
                          size_t kh, size_t kw,
                          size_t sh, size_t sw,
                          size_t ph, size_t pw,
                          FloatTensor *out) {
  const size_t N = in->shape[0], C = in->shape[1];
  const size_t H = in->shape[2], W = in->shape[3];
  const size_t OH = out->shape[2], OW = out->shape[3];
  for (size_t n = 0; n < N; ++n)
  for (size_t c = 0; c < C; ++c)
  for (size_t oh = 0; oh < OH; ++oh)
  for (size_t ow = 0; ow < OW; ++ow) {
    float mx = -FLT_MAX;
    for (size_t ki = 0; ki < kh; ++ki)
    for (size_t kj = 0; kj < kw; ++kj) {
      size_t ih = oh * sh + ki;
      size_t iw = ow * sw + kj;
      if (ih < ph || ih - ph >= H) continue;
      if (iw < pw || iw - pw >= W) continue;
      size_t idx = ((n * C + c) * H + (ih - ph)) * W + (iw - pw);
      if (in->values[idx] > mx) mx = in->values[idx];
    }
    out->values[((n * C + c) * OH + oh) * OW + ow] = mx;
  }
}

/* Inline reference: AvgPool2D (NCHW, exclude-padding mode). */
static void ref_avgpool2d(const FloatTensor *in,
                          size_t kh, size_t kw,
                          size_t sh, size_t sw,
                          size_t ph, size_t pw,
                          FloatTensor *out) {
  const size_t N = in->shape[0], C = in->shape[1];
  const size_t H = in->shape[2], W = in->shape[3];
  const size_t OH = out->shape[2], OW = out->shape[3];
  for (size_t n = 0; n < N; ++n)
  for (size_t c = 0; c < C; ++c)
  for (size_t oh = 0; oh < OH; ++oh)
  for (size_t ow = 0; ow < OW; ++ow) {
    float sum = 0.0f;
    size_t cnt = 0;
    for (size_t ki = 0; ki < kh; ++ki)
    for (size_t kj = 0; kj < kw; ++kj) {
      size_t ih = oh * sh + ki;
      size_t iw = ow * sw + kj;
      if (ih < ph || ih - ph >= H) continue;
      if (iw < pw || iw - pw >= W) continue;
      size_t idx = ((n * C + c) * H + (ih - ph)) * W + (iw - pw);
      sum += in->values[idx];
      ++cnt;
    }
    out->values[((n * C + c) * OH + oh) * OW + ow] =
        cnt > 0 ? sum / (float)cnt : 0.0f;
  }
}

/* Inline reference: BatchNorm2D forward (NCHW). */
static void ref_batchnorm2d(const FloatTensor *in,
                            const FloatTensor *gamma,
                            const FloatTensor *beta,
                            float eps,
                            FloatTensor *out) {
  const size_t N = in->shape[0], C = in->shape[1];
  const size_t H = in->shape[2], W = in->shape[3];
  const size_t HW = H * W;
  for (size_t c = 0; c < C; ++c) {
    /* compute mean */
    double sum = 0.0;
    for (size_t n = 0; n < N; ++n)
    for (size_t hw = 0; hw < HW; ++hw)
      sum += in->values[(n * C + c) * HW + hw];
    float mean = (float)(sum / (double)(N * HW));
    /* compute variance */
    double var_sum = 0.0;
    for (size_t n = 0; n < N; ++n)
    for (size_t hw = 0; hw < HW; ++hw) {
      float d = in->values[(n * C + c) * HW + hw] - mean;
      var_sum += (double)(d * d);
    }
    float inv_std = 1.0f / sqrtf((float)(var_sum / (double)(N * HW)) + eps);
    float g = gamma ? gamma->values[c] : 1.0f;
    float b = beta  ? beta->values[c]  : 0.0f;
    for (size_t n = 0; n < N; ++n)
    for (size_t hw = 0; hw < HW; ++hw) {
      size_t idx = (n * C + c) * HW + hw;
      out->values[idx] = (in->values[idx] - mean) * inv_std * g + b;
    }
  }
}

static void print_counters(void) {
  StBackendCounters c = st_backend_get_counters();
  printf("    counters: mps_hit=%ld  mps_miss=%ld  fallback_gemm=%ld"
         "  fallback_ref=%ld\n",
         c.mps_hit, c.mps_miss, c.fallback_gemm, c.fallback_ref);
}

/* ------------------------------------------------------------------ */
/*  Conv2D (AUTO backend)                                              */
/* ------------------------------------------------------------------ */

typedef struct ConvCase {
  const char *name;
  size_t n, c_in, c_out, h, w, k, stride, pad;
  size_t warmup, iters;
} ConvCase;

static void bench_conv2d(const ConvCase *cfg) {
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

  FloatTensor *input  = make4d(cfg->n, cfg->c_in, cfg->h, cfg->w);
  FloatTensor *weight = make4d(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
  FloatTensor *output = make4d(cfg->n, cfg->c_out, out_h, out_w);
  if (!input || !weight || !output) {
    fprintf(stderr, "  [OOM] %s\n", cfg->name);
    st_destroy(input); st_destroy(weight); st_destroy(output);
    return;
  }
  fill_rand(input, 1u);
  fill_rand(weight, 2u);

  /* warmup */
  for (size_t i = 0; i < cfg->warmup; ++i)
    st_conv2d_nchw(input, weight, NULL, &p, output);

  st_backend_reset_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < cfg->iters; ++i)
    st_conv2d_nchw(input, weight, NULL, &p, output);
  uint64_t t1 = now_ns();

  StBackendCounters c = st_backend_get_counters();
  double ms = elapsed_ms(t0, t1, cfg->iters);
  FloatTensor *ref_out = make4d(cfg->n, cfg->c_out, out_h, out_w);
  float max_abs = 0.0f, max_rel = 0.0f;
  if (ref_out) {
    StConv2dParams pr = p;
    pr.backend = ST_CONV_BACKEND_GEMM;
    st_conv2d_nchw(input, weight, NULL, &pr, ref_out);
    accuracy(output, ref_out, &max_abs, &max_rel);
    st_destroy(ref_out);
  }
  if (bench_csv_enabled()) {
    print_csv_row("conv2d", cfg->name, cfg->n, cfg->c_in, cfg->c_out, cfg->h, cfg->w, cfg->k, cfg->stride, cfg->pad, ms, c, max_abs, max_rel);
  } else {
    printf("[single-op] conv2d  %s\n", cfg->name);
    printf("    N=%zu Cin=%zu Cout=%zu H=%zu W=%zu K=%zu s=%zu p=%zu\n",
           cfg->n, cfg->c_in, cfg->c_out, cfg->h, cfg->w, cfg->k, cfg->stride, cfg->pad);
    printf("    conv2d_auto: %.3f ms/iter\n", ms);
    print_counters();
    print_accuracy(max_abs, max_rel);
  }

  st_destroy(input);
  st_destroy(weight);
  st_destroy(output);
}

/* ------------------------------------------------------------------ */
/*  MaxPool2D                                                          */
/* ------------------------------------------------------------------ */

typedef struct PoolCase {
  const char *name;
  size_t n, c, h, w, kh, kw, sh, sw, ph, pw;
  size_t warmup, iters;
} PoolCase;

static void bench_maxpool2d(const PoolCase *cfg) {
  size_t oh = (cfg->h + 2 * cfg->ph - cfg->kh) / cfg->sh + 1;
  size_t ow = (cfg->w + 2 * cfg->pw - cfg->kw) / cfg->sw + 1;

  FloatTensor *input  = make4d(cfg->n, cfg->c, cfg->h, cfg->w);
  FloatTensor *output = make4d(cfg->n, cfg->c, oh, ow);
  if (!input || !output) {
    fprintf(stderr, "  [OOM] %s\n", cfg->name);
    st_destroy(input); st_destroy(output);
    return;
  }
  fill_rand(input, 3u);

  for (size_t i = 0; i < cfg->warmup; ++i)
    st_maxpool2d_nchw(input, cfg->kh, cfg->kw, cfg->sh, cfg->sw,
                      cfg->ph, cfg->pw, output, NULL);

  st_backend_reset_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < cfg->iters; ++i)
    st_maxpool2d_nchw(input, cfg->kh, cfg->kw, cfg->sh, cfg->sw,
                      cfg->ph, cfg->pw, output, NULL);
  uint64_t t1 = now_ns();

  StBackendCounters c = st_backend_get_counters();
  double ms = elapsed_ms(t0, t1, cfg->iters);
  FloatTensor *ref_out = make4d(cfg->n, cfg->c, oh, ow);
  float max_abs = 0.0f, max_rel = 0.0f;
  if (ref_out) {
    ref_maxpool2d(input, cfg->kh, cfg->kw, cfg->sh, cfg->sw, cfg->ph, cfg->pw, ref_out);
    accuracy(output, ref_out, &max_abs, &max_rel);
    st_destroy(ref_out);
  }
  if (bench_csv_enabled()) {
    print_csv_row("maxpool2d", cfg->name, cfg->n, cfg->c, 0, cfg->h, cfg->w, cfg->kh, cfg->sh, cfg->ph, ms, c, max_abs, max_rel);
  } else {
    printf("[single-op] maxpool2d  %s\n", cfg->name);
    printf("    N=%zu C=%zu H=%zu W=%zu k=%zux%zu s=%zux%zu p=%zux%zu\n",
           cfg->n, cfg->c, cfg->h, cfg->w, cfg->kh, cfg->kw, cfg->sh, cfg->sw, cfg->ph, cfg->pw);
    printf("    maxpool2d: %.3f ms/iter\n", ms);
    print_counters();
    print_accuracy(max_abs, max_rel);
  }

  st_destroy(input);
  st_destroy(output);
}

/* ------------------------------------------------------------------ */
/*  AvgPool2D                                                          */
/* ------------------------------------------------------------------ */

static void bench_avgpool2d(const PoolCase *cfg) {
  size_t oh = (cfg->h + 2 * cfg->ph - cfg->kh) / cfg->sh + 1;
  size_t ow = (cfg->w + 2 * cfg->pw - cfg->kw) / cfg->sw + 1;

  FloatTensor *input  = make4d(cfg->n, cfg->c, cfg->h, cfg->w);
  FloatTensor *output = make4d(cfg->n, cfg->c, oh, ow);
  if (!input || !output) {
    fprintf(stderr, "  [OOM] %s\n", cfg->name);
    st_destroy(input); st_destroy(output);
    return;
  }
  fill_rand(input, 4u);

  for (size_t i = 0; i < cfg->warmup; ++i)
    st_avgpool2d_nchw(input, cfg->kh, cfg->kw, cfg->sh, cfg->sw,
                      cfg->ph, cfg->pw, output);

  st_backend_reset_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < cfg->iters; ++i)
    st_avgpool2d_nchw(input, cfg->kh, cfg->kw, cfg->sh, cfg->sw,
                      cfg->ph, cfg->pw, output);
  uint64_t t1 = now_ns();

  StBackendCounters c = st_backend_get_counters();
  double ms = elapsed_ms(t0, t1, cfg->iters);
  FloatTensor *ref_out = make4d(cfg->n, cfg->c, oh, ow);
  float max_abs = 0.0f, max_rel = 0.0f;
  if (ref_out) {
    ref_avgpool2d(input, cfg->kh, cfg->kw, cfg->sh, cfg->sw, cfg->ph, cfg->pw, ref_out);
    accuracy(output, ref_out, &max_abs, &max_rel);
    st_destroy(ref_out);
  }
  if (bench_csv_enabled()) {
    print_csv_row("avgpool2d", cfg->name, cfg->n, cfg->c, 0, cfg->h, cfg->w, cfg->kh, cfg->sh, cfg->ph, ms, c, max_abs, max_rel);
  } else {
    printf("[single-op] avgpool2d  %s\n", cfg->name);
    printf("    N=%zu C=%zu H=%zu W=%zu k=%zux%zu s=%zux%zu p=%zux%zu\n",
           cfg->n, cfg->c, cfg->h, cfg->w, cfg->kh, cfg->kw, cfg->sh, cfg->sw, cfg->ph, cfg->pw);
    printf("    avgpool2d: %.3f ms/iter\n", ms);
    print_counters();
    print_accuracy(max_abs, max_rel);
  }

  st_destroy(input);
  st_destroy(output);
}

/* ------------------------------------------------------------------ */
/*  BatchNorm2D                                                        */
/* ------------------------------------------------------------------ */

typedef struct BNCase {
  const char *name;
  size_t n, c, h, w;
  size_t warmup, iters;
} BNCase;

static void bench_batchnorm2d(const BNCase *cfg) {
  FloatTensor *input  = make4d(cfg->n, cfg->c, cfg->h, cfg->w);
  FloatTensor *output = make4d(cfg->n, cfg->c, cfg->h, cfg->w);
  FloatTensor *gamma  = make1d(cfg->c);
  FloatTensor *beta   = make1d(cfg->c);
  FloatTensor *mean   = make1d(cfg->c);
  FloatTensor *var    = make1d(cfg->c);
  if (!input || !output || !gamma || !beta || !mean || !var) {
    fprintf(stderr, "  [OOM] %s\n", cfg->name);
    st_destroy(input); st_destroy(output); st_destroy(gamma);
    st_destroy(beta);  st_destroy(mean);   st_destroy(var);
    return;
  }
  fill_rand(input, 5u);
  fill_rand(gamma, 6u);
  fill_rand(beta,  7u);

  for (size_t i = 0; i < cfg->warmup; ++i)
    st_batchnorm2d_forward(input, gamma, beta, 1e-5f, output, mean, var);

  st_backend_reset_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < cfg->iters; ++i)
    st_batchnorm2d_forward(input, gamma, beta, 1e-5f, output, mean, var);
  uint64_t t1 = now_ns();

  StBackendCounters c = st_backend_get_counters();
  double ms = elapsed_ms(t0, t1, cfg->iters);
  FloatTensor *ref_out = make4d(cfg->n, cfg->c, cfg->h, cfg->w);
  float max_abs = 0.0f, max_rel = 0.0f;
  if (ref_out) {
    ref_batchnorm2d(input, gamma, beta, 1e-5f, ref_out);
    accuracy(output, ref_out, &max_abs, &max_rel);
    st_destroy(ref_out);
  }
  if (bench_csv_enabled()) {
    print_csv_row("batchnorm2d", cfg->name, cfg->n, cfg->c, 0, cfg->h, cfg->w, 0, 0, 0, ms, c, max_abs, max_rel);
  } else {
    printf("[single-op] batchnorm2d  %s\n", cfg->name);
    printf("    N=%zu C=%zu H=%zu W=%zu\n", cfg->n, cfg->c, cfg->h, cfg->w);
    printf("    batchnorm2d: %.3f ms/iter\n", ms);
    print_counters();
    print_accuracy(max_abs, max_rel);
  }

  st_destroy(input);  st_destroy(output);
  st_destroy(gamma);  st_destroy(beta);
  st_destroy(mean);   st_destroy(var);
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
  if (bench_csv_enabled()) print_csv_header();
  else printf("=== bench_st_single_op ===\n\n");

  /* ---- Conv2D cases ---- */
  static const ConvCase conv_cases[] = {
    /* small: below MPS threshold → CPU path */
    { "conv-small",  1,  8,  8, 16, 16, 3, 1, 1, 2, 20 },
    /* medium: borderline */
    { "conv-medium", 4, 32, 64, 56, 56, 3, 1, 1, 2, 10 },
    /* large: above MPS threshold → MPS on Apple Silicon */
    { "conv-large",  8, 64,128,112,112, 3, 1, 1, 2,  5 },
  };
  for (size_t i = 0; i < sizeof(conv_cases)/sizeof(conv_cases[0]); ++i)
    bench_conv2d(&conv_cases[i]);

  printf("\n");

  /* ---- MaxPool2D cases ---- */
  static const PoolCase pool_cases[] = {
    { "pool-small",  1, 16, 32, 32, 2, 2, 2, 2, 0, 0, 2, 50 },
    { "pool-large",  8, 64,112,112, 3, 3, 2, 2, 1, 1, 2, 10 },
  };
  for (size_t i = 0; i < sizeof(pool_cases)/sizeof(pool_cases[0]); ++i)
    bench_maxpool2d(&pool_cases[i]);

  printf("\n");

  /* ---- AvgPool2D cases ---- */
  static const PoolCase avgpool_cases[] = {
    { "avgpool-small", 1, 16, 32, 32, 2, 2, 2, 2, 0, 0, 2, 50 },
    { "avgpool-large", 8, 64, 56, 56, 3, 3, 1, 1, 1, 1, 2, 10 },
  };
  for (size_t i = 0; i < sizeof(avgpool_cases)/sizeof(avgpool_cases[0]); ++i)
    bench_avgpool2d(&avgpool_cases[i]);

  printf("\n");

  /* ---- BatchNorm2D cases ---- */
  static const BNCase bn_cases[] = {
    { "bn-small", 4,  32, 28, 28, 2, 50 },
    { "bn-large", 8, 128, 56, 56, 2, 10 },
  };
  for (size_t i = 0; i < sizeof(bn_cases)/sizeof(bn_cases[0]); ++i)
    bench_batchnorm2d(&bn_cases[i]);

  if (!bench_csv_enabled()) {
    printf("\n=== done ===\n");
  }
  return 0;
}
