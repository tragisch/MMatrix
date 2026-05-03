#include "st.h"
#include "st_backend.h"
#include "st_batchnorm.h"
#include "st_conv.h"
#include "st_pool.h"
#include "log.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct BenchCountersDelta {
  long mps_hit;
  long mps_miss;
  long fallback_gemm;
  long fallback_ref;
} BenchCountersDelta;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double elapsed_ms(uint64_t start, uint64_t end, size_t iters) {
  return (double)(end - start) / 1000000.0 / (double)iters;
}

static FloatTensor *make4d(size_t n, size_t c, size_t h, size_t w) {
  size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

static FloatTensor *make1d(size_t n) {
  size_t shape[1] = {n};
  return st_create(1, shape);
}

static void fill_rand(FloatTensor *t, uint32_t seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1664525u + 1013904223u;
    t->values[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

static float max_abs_diff(const FloatTensor *a, const FloatTensor *b) {
  float max_abs = 0.0f;
  if (!a || !b || a->numel != b->numel) {
    return -1.0f;
  }
  for (size_t i = 0; i < a->numel; ++i) {
    float diff = fabsf(a->values[i] - b->values[i]);
    if (diff > max_abs) {
      max_abs = diff;
    }
  }
  return max_abs;
}

static BenchCountersDelta counters_delta(StBackendCounters before,
                                         StBackendCounters after) {
  BenchCountersDelta delta;
  delta.mps_hit = after.mps_hit - before.mps_hit;
  delta.mps_miss = after.mps_miss - before.mps_miss;
  delta.fallback_gemm = after.fallback_gemm - before.fallback_gemm;
  delta.fallback_ref = after.fallback_ref - before.fallback_ref;
  return delta;
}

static const char *fallback_reason_name(void) {
  return st_fallback_reason_str(st_get_last_fallback_reason());
}

static const char *path_name_from_counters(BenchCountersDelta delta) {
  if (delta.mps_hit > 0 && delta.fallback_gemm > 0 && delta.mps_miss > 0) {
    return "mixed_mps_and_fallback_gemm";
  }
  if (delta.mps_hit > 0 && delta.fallback_gemm > 0) {
    return "mixed_mps_and_gemm";
  }
  if (delta.mps_hit > 0 && delta.fallback_ref > 0 && delta.mps_miss > 0) {
    return "mixed_mps_and_fallback_ref";
  }
  if (delta.mps_hit > 0 && delta.fallback_ref > 0) {
    return "mixed_mps_and_ref";
  }
  if (delta.mps_hit > 0 && delta.mps_miss == 0 &&
      delta.fallback_gemm == 0 && delta.fallback_ref == 0) {
    return "mps";
  }
  if (delta.mps_hit == 0 && delta.mps_miss == 0 &&
      delta.fallback_gemm == 0 && delta.fallback_ref == 0) {
    return "cpu_direct";
  }
  if (delta.fallback_gemm > 0 && delta.mps_miss > 0) {
    return "mps_fallback_gemm";
  }
  if (delta.fallback_ref > 0 && delta.mps_miss > 0) {
    return "mps_fallback_ref";
  }
  if (delta.fallback_gemm > 0) {
    return "cpu_gemm";
  }
  if (delta.fallback_ref > 0) {
    return "cpu_ref";
  }
  if (delta.mps_miss > 0) {
    return "mps_miss";
  }
  return "mixed";
}

static void print_header(void) {
  printf("suite,case_name,variant,requested_backend,observed_backend,path,"
         "fallback_reason,warmup,iters,ms_per_iter,mps_hit,mps_miss,"
         "fallback_gemm,fallback_ref,max_abs_diff\n");
}

static void print_row(const char *suite, const char *case_name,
                      const char *variant, const char *requested_backend,
                      const char *observed_backend, const char *path,
                      const char *fallback_reason, size_t warmup,
                      size_t iters, double ms_per_iter,
                      BenchCountersDelta delta, float max_abs) {
  printf("%s,%s,%s,%s,%s,%s,%s,%zu,%zu,%.6f,%ld,%ld,%ld,%ld,",
         suite, case_name, variant, requested_backend, observed_backend,
         path, fallback_reason, warmup, iters, ms_per_iter,
         delta.mps_hit, delta.mps_miss, delta.fallback_gemm,
         delta.fallback_ref);
  if (max_abs >= 0.0f) {
    printf("%.8f\n", (double)max_abs);
  } else {
    printf("na\n");
  }
}

static int bench_conv_case(const char *case_name,
                           size_t n, size_t c_in, size_t c_out,
                           size_t h, size_t w, size_t k,
                           size_t stride, size_t pad,
                           size_t warmup, size_t iters) {
  int rc = 1;
  size_t out_h = 0;
  size_t out_w = 0;
  StConv2dParams params = st_conv2d_default_params();
  params.stride_h = stride;
  params.stride_w = stride;
  params.pad_h = pad;
  params.pad_w = pad;
  params.backend = ST_CONV_BACKEND_AUTO;

  if (!st_conv2d_output_hw(h, w, k, k, &params, &out_h, &out_w)) {
    return 1;
  }

  FloatTensor *input = make4d(n, c_in, h, w);
  FloatTensor *weight = make4d(c_out, c_in, k, k);
  FloatTensor *output = make4d(n, c_out, out_h, out_w);
  FloatTensor *ref = make4d(n, c_out, out_h, out_w);
  if (!input || !weight || !output || !ref) {
    goto cleanup;
  }

  fill_rand(input, 1u);
  fill_rand(weight, 2u);

  for (size_t i = 0; i < warmup; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, &params, output)) {
      goto cleanup;
    }
  }

  StBackendCounters before = st_backend_get_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < iters; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, &params, output)) {
      goto cleanup;
    }
  }
  st_tensor_sync(output);
  uint64_t t1 = now_ns();
  StBackendCounters after = st_backend_get_counters();
  BenchCountersDelta delta = counters_delta(before, after);
  const char *observed_backend = st_conv2d_last_backend();
  const char *fallback_reason = fallback_reason_name();

  StConv2dParams ref_params = params;
  ref_params.backend = ST_CONV_BACKEND_GEMM;
  if (!st_conv2d_nchw(input, weight, NULL, &ref_params, ref)) {
    goto cleanup;
  }

  print_row("conv2d", case_name, "auto", "auto", observed_backend,
            path_name_from_counters(delta), fallback_reason, warmup, iters,
            elapsed_ms(t0, t1, iters), delta, max_abs_diff(output, ref));
  rc = 0;

cleanup:
  st_destroy(ref);
  st_destroy(output);
  st_destroy(weight);
  st_destroy(input);
  return rc;
}

static int bench_maxpool_case(const char *case_name,
                              size_t n, size_t c, size_t h, size_t w,
                              size_t kh, size_t kw,
                              size_t sh, size_t sw,
                              size_t ph, size_t pw,
                              size_t warmup, size_t iters) {
  int rc = 1;
  size_t out_h = 0;
  size_t out_w = 0;
  if (!st_pool2d_output_hw(h, w, kh, kw, sh, sw, ph, pw, &out_h, &out_w)) {
    return 1;
  }

  FloatTensor *input = make4d(n, c, h, w);
  FloatTensor *output = make4d(n, c, out_h, out_w);
  if (!input || !output) {
    goto cleanup;
  }
  fill_rand(input, 3u);

  for (size_t i = 0; i < warmup; ++i) {
    if (!st_maxpool2d_nchw(input, kh, kw, sh, sw, ph, pw, output, NULL)) {
      goto cleanup;
    }
  }

  StBackendCounters before = st_backend_get_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < iters; ++i) {
    if (!st_maxpool2d_nchw(input, kh, kw, sh, sw, ph, pw, output, NULL)) {
      goto cleanup;
    }
  }
  st_tensor_sync(output);
  uint64_t t1 = now_ns();
  StBackendCounters after = st_backend_get_counters();
  BenchCountersDelta delta = counters_delta(before, after);

  print_row("maxpool2d", case_name, "forward", "auto", "pool_dispatch",
            path_name_from_counters(delta), fallback_reason_name(), warmup,
            iters, elapsed_ms(t0, t1, iters), delta, -1.0f);
  rc = 0;

cleanup:
  st_destroy(output);
  st_destroy(input);
  return rc;
}

static int bench_batchnorm_case(const char *case_name,
                                size_t n, size_t c, size_t h, size_t w,
                                size_t warmup, size_t iters) {
  int rc = 1;
  FloatTensor *input = make4d(n, c, h, w);
  FloatTensor *output = make4d(n, c, h, w);
  FloatTensor *mean = make1d(c);
  FloatTensor *var = make1d(c);
  if (!input || !output || !mean || !var) {
    goto cleanup;
  }
  fill_rand(input, 4u);

  for (size_t i = 0; i < warmup; ++i) {
    if (!st_batchnorm2d_forward(input, NULL, NULL, 1e-5f, output, mean, var)) {
      goto cleanup;
    }
  }

  StBackendCounters before = st_backend_get_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < iters; ++i) {
    if (!st_batchnorm2d_forward(input, NULL, NULL, 1e-5f, output, mean, var)) {
      goto cleanup;
    }
  }
  st_tensor_sync(output);
  uint64_t t1 = now_ns();
  StBackendCounters after = st_backend_get_counters();
  BenchCountersDelta delta = counters_delta(before, after);

  print_row("batchnorm2d", case_name, "forward", "auto", "batchnorm_dispatch",
            path_name_from_counters(delta), fallback_reason_name(), warmup,
            iters, elapsed_ms(t0, t1, iters), delta, -1.0f);
  rc = 0;

cleanup:
  st_destroy(var);
  st_destroy(mean);
  st_destroy(output);
  st_destroy(input);
  return rc;
}

static int bench_conv_bn_case(const char *case_name,
                              size_t n, size_t c_in, size_t c_out,
                              size_t h, size_t w, size_t k,
                              size_t stride, size_t pad,
                              size_t warmup, size_t iters) {
  int rc = 1;
  size_t out_h = 0;
  size_t out_w = 0;
  StConv2dParams params = st_conv2d_default_params();
  params.stride_h = stride;
  params.stride_w = stride;
  params.pad_h = pad;
  params.pad_w = pad;
  params.backend = ST_CONV_BACKEND_AUTO;

  if (!st_conv2d_output_hw(h, w, k, k, &params, &out_h, &out_w)) {
    return 1;
  }

  FloatTensor *input = make4d(n, c_in, h, w);
  FloatTensor *weight = make4d(c_out, c_in, k, k);
  FloatTensor *sep_conv = make4d(n, c_out, out_h, out_w);
  FloatTensor *sep_out = make4d(n, c_out, out_h, out_w);
  FloatTensor *fused_out = make4d(n, c_out, out_h, out_w);
  FloatTensor *gamma = make1d(c_out);
  FloatTensor *beta = make1d(c_out);
  FloatTensor *mean = make1d(c_out);
  FloatTensor *var = make1d(c_out);
  FloatTensor *mean_fused = make1d(c_out);
  FloatTensor *var_fused = make1d(c_out);
  if (!input || !weight || !sep_conv || !sep_out || !fused_out ||
      !gamma || !beta || !mean || !var || !mean_fused || !var_fused) {
    goto cleanup;
  }

  fill_rand(input, 10u);
  fill_rand(weight, 20u);
  fill_rand(gamma, 30u);
  fill_rand(beta, 40u);

  for (size_t i = 0; i < warmup; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, &params, sep_conv)) {
      goto cleanup;
    }
    if (!st_batchnorm2d_forward(sep_conv, gamma, beta, 1e-5f,
                                sep_out, mean, var)) {
      goto cleanup;
    }
    if (!st_conv2d_batchnorm2d_forward_nchw(input, weight, NULL, &params,
                                            gamma, beta, 1e-5f, fused_out,
                                            mean_fused, var_fused, false)) {
      goto cleanup;
    }
  }

  StBackendCounters before_sep = st_backend_get_counters();
  StBackendFallbackReason sep_conv_reason = ST_FALLBACK_NONE;
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < iters; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, &params, sep_conv)) {
      goto cleanup;
    }
    sep_conv_reason = st_get_last_fallback_reason();
    if (!st_batchnorm2d_forward(sep_conv, gamma, beta, 1e-5f,
                                sep_out, mean, var)) {
      goto cleanup;
    }
  }
  st_tensor_sync(sep_out);
  uint64_t t1 = now_ns();
  StBackendCounters after_sep = st_backend_get_counters();
  BenchCountersDelta delta_sep = counters_delta(before_sep, after_sep);
  const char *sep_backend = st_conv2d_last_backend();
  const char *sep_reason = st_fallback_reason_str(sep_conv_reason);

  StBackendCounters before_fused = st_backend_get_counters();
  uint64_t t2 = now_ns();
  for (size_t i = 0; i < iters; ++i) {
    if (!st_conv2d_batchnorm2d_forward_nchw(input, weight, NULL, &params,
                                            gamma, beta, 1e-5f, fused_out,
                                            mean_fused, var_fused, false)) {
      goto cleanup;
    }
  }
  st_tensor_sync(fused_out);
  uint64_t t3 = now_ns();
  StBackendCounters after_fused = st_backend_get_counters();
  BenchCountersDelta delta_fused = counters_delta(before_fused, after_fused);
  const char *fused_backend = st_conv2d_last_backend();
  const char *fused_reason = fallback_reason_name();
  if (strcmp(fused_backend, "mps_conv_bn") != 0 &&
      strcmp(fused_backend, "mps_conv_bn_relu") != 0) {
    if (!st_conv2d_nchw(input, weight, NULL, &params, sep_conv)) {
      goto cleanup;
    }
    fused_reason = fallback_reason_name();
  }

  if (!st_conv2d_nchw(input, weight, NULL, &params, sep_conv)) {
    goto cleanup;
  }
  if (!st_batchnorm2d_forward(sep_conv, gamma, beta, 1e-5f,
                              sep_out, mean, var)) {
    goto cleanup;
  }
  if (!st_conv2d_batchnorm2d_forward_nchw(input, weight, NULL, &params,
                                          gamma, beta, 1e-5f, fused_out,
                                          mean_fused, var_fused, false)) {
    goto cleanup;
  }

  print_row("conv_bn", case_name, "separate", "auto", sep_backend,
            path_name_from_counters(delta_sep), sep_reason, warmup, iters,
            elapsed_ms(t0, t1, iters), delta_sep, -1.0f);
  print_row("conv_bn", case_name, "fused", "auto", fused_backend,
            path_name_from_counters(delta_fused), fused_reason, warmup, iters,
            elapsed_ms(t2, t3, iters), delta_fused,
            max_abs_diff(sep_out, fused_out));
  rc = 0;

cleanup:
  st_destroy(var_fused);
  st_destroy(mean_fused);
  st_destroy(var);
  st_destroy(mean);
  st_destroy(beta);
  st_destroy(gamma);
  st_destroy(fused_out);
  st_destroy(sep_out);
  st_destroy(sep_conv);
  st_destroy(weight);
  st_destroy(input);
  return rc;
}

int main(void) {
  log_set_level(LOG_INFO);
  print_header();

  if (bench_conv_case("conv_small", 1, 8, 8, 16, 16, 3, 1, 1, 2, 20) != 0) {
    return 1;
  }
  if (bench_conv_case("conv_medium", 4, 32, 64, 56, 56, 3, 1, 1, 2, 10) != 0) {
    return 1;
  }
  if (bench_conv_case("conv_large", 8, 64, 128, 112, 112, 3, 1, 1, 2, 5) != 0) {
    return 1;
  }
  if (bench_maxpool_case("pool_large", 8, 64, 112, 112, 3, 3, 2, 2, 1, 1,
                         2, 10) != 0) {
    return 1;
  }
  if (bench_batchnorm_case("bn_large", 8, 128, 56, 56, 2, 10) != 0) {
    return 1;
  }
  if (bench_conv_bn_case("pipe_medium", 4, 32, 64, 56, 56, 3, 1, 1, 2, 10) != 0) {
    return 1;
  }
  if (bench_conv_bn_case("pipe_large", 8, 64, 128, 112, 112, 3, 1, 1, 2, 5) != 0) {
    return 1;
  }

  return 0;
}
