#include "st.h"
#include "st_backend.h"
#include "st_conv.h"

#include <math.h>
#include <stdbool.h>
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

static void fill_rand(FloatTensor *t, uint32_t seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1664525u + 1013904223u;
    t->values[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

static float max_abs_diff(const FloatTensor *a, const FloatTensor *b) {
  float max_abs = 0.0f;
  if (!a || !b || a->numel != b->numel) return -1.0f;
  for (size_t i = 0; i < a->numel; ++i) {
    float d = fabsf(a->values[i] - b->values[i]);
    if (d > max_abs) max_abs = d;
  }
  return max_abs;
}

static BenchCountersDelta counters_delta(StBackendCounters before,
                                         StBackendCounters after) {
  return (BenchCountersDelta){
      .mps_hit = after.mps_hit - before.mps_hit,
      .mps_miss = after.mps_miss - before.mps_miss,
      .fallback_gemm = after.fallback_gemm - before.fallback_gemm,
      .fallback_ref = after.fallback_ref - before.fallback_ref,
  };
}

static void print_header(void) {
  printf("suite,case_name,variant,phase,requested_backend,observed_backend,"
         "iters,ms_per_iter,mps_hit,mps_miss,fallback_gemm,fallback_ref,"
         "max_abs_diff,out_buf_type,readbytes_delta,fastpath_delta,"
         "fp_exec_nil_delta,fp_missing_feed_delta,fp_preout_nil_delta,"
         "fp_cmd_buf_nil_delta,fp_encode_exc_delta\n");
}

static void print_row(const char *case_name, const char *variant,
                      const char *phase, const char *requested_backend,
                      const char *observed_backend, size_t iters, double ms,
                      BenchCountersDelta d, float diff,
                      const char *out_buf_type, long readbytes_delta,
                      long fastpath_delta, long fp_exec_nil_delta,
                      long fp_missing_feed_delta, long fp_preout_nil_delta,
                      long fp_cmd_buf_nil_delta, long fp_encode_exc_delta) {
  printf("conv_medium_profile,%s,%s,%s,%s,%s,%zu,%.6f,%ld,%ld,%ld,%ld,",
         case_name, variant, phase, requested_backend, observed_backend, iters,
         ms, d.mps_hit, d.mps_miss, d.fallback_gemm, d.fallback_ref);
  if (diff >= 0.0f) {
    printf("%.8f", (double)diff);
  } else {
    printf("na");
  }
  printf(",%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld\n",
         out_buf_type, readbytes_delta, fastpath_delta, fp_exec_nil_delta,
         fp_missing_feed_delta, fp_preout_nil_delta, fp_cmd_buf_nil_delta,
         fp_encode_exc_delta);
}

static int run_phase(const char *case_name, const char *variant,
                     const char *phase, const StConv2dParams *params,
                     const FloatTensor *input, const FloatTensor *weight,
                     const FloatTensor *ref, FloatTensor *out, size_t iters,
                     bool sync_each_iter, bool sync_at_end) {
  const bool out_is_metal =
      (out->buf && st_buffer_metal_handle(out->buf) != NULL);
  const char *out_buf_type = out_is_metal ? "metal" : "cpu";

  StBackendCounters c0 = st_backend_get_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < iters; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, params, out)) {
      return 1;
    }
    if (sync_each_iter) {
      st_tensor_sync(out);
    }
  }
  if (sync_at_end) {
    st_tensor_sync(out);
  }
  uint64_t t1 = now_ns();
  StBackendCounters c1 = st_backend_get_counters();

  BenchCountersDelta d = counters_delta(c0, c1);
  const long readbytes_delta = c1.conv_readbytes - c0.conv_readbytes;
  const long fastpath_delta = c1.conv_fastpath_hit - c0.conv_fastpath_hit;
  const long fp_exec_nil_delta =
      c1.conv_fastpath_executable_nil - c0.conv_fastpath_executable_nil;
  const long fp_missing_feed_delta =
      c1.conv_fastpath_missing_feed - c0.conv_fastpath_missing_feed;
  const long fp_preout_nil_delta =
      c1.conv_fastpath_preout_nil - c0.conv_fastpath_preout_nil;
  const long fp_cmd_buf_nil_delta =
      c1.conv_fastpath_cmd_buf_nil - c0.conv_fastpath_cmd_buf_nil;
  const long fp_encode_exc_delta =
      c1.conv_fastpath_encode_exception - c0.conv_fastpath_encode_exception;

  print_row(case_name, variant, phase, "mps", st_conv2d_last_backend(), iters,
            elapsed_ms(t0, t1, iters), d, max_abs_diff(ref, out), out_buf_type,
            readbytes_delta, fastpath_delta, fp_exec_nil_delta,
            fp_missing_feed_delta, fp_preout_nil_delta, fp_cmd_buf_nil_delta,
            fp_encode_exc_delta);
  return 0;
}

static int run_variant(const char *case_name, const char *variant,
                       const StConv2dParams *mps_params,
                       const FloatTensor *input, const FloatTensor *weight,
                       const FloatTensor *ref, FloatTensor *out,
                       bool sync_each_iter, bool sync_at_end) {
  if (run_phase(case_name, variant, "first_call", mps_params,
                input, weight, ref, out, 1, sync_each_iter, sync_at_end) != 0) {
    return 1;
  }
  if (run_phase(case_name, variant, "second_call", mps_params,
                input, weight, ref, out, 1, sync_each_iter, sync_at_end) != 0) {
    return 1;
  }
  if (run_phase(case_name, variant, "steady_state", mps_params,
                input, weight, ref, out, 20, sync_each_iter, sync_at_end) != 0) {
    return 1;
  }
  return 0;
}

static bool should_run_variant(const char *requested, const char *variant) {
  return requested == NULL || strcmp(requested, "all") == 0 ||
         strcmp(requested, variant) == 0;
}

int main(int argc, char **argv) {
  int rc = 1;
  const char *requested_variant = NULL;
  const char *case_name = "conv_medium";
  const size_t n = 4, c_in = 32, c_out = 64, h = 56, w = 56, k = 3;
  const size_t stride = 1, pad = 1;
  size_t out_h = 0, out_w = 0;

  if (argc > 2) {
    fprintf(stderr, "usage: %s [all|mps_zero_copy_sync|mps_true_async_boundary]\n",
            argv[0]);
    return 2;
  }
  if (argc == 2) {
    requested_variant = argv[1];
    if (strcmp(requested_variant, "all") != 0 &&
        strcmp(requested_variant, "mps_zero_copy_sync") != 0 &&
        strcmp(requested_variant, "mps_true_async_boundary") != 0) {
      fprintf(stderr, "unknown variant: %s\n", requested_variant);
      return 2;
    }
  }

  StConv2dParams base = st_conv2d_default_params();
  base.backend = ST_CONV_BACKEND_MPS;
  base.stride_h = stride;
  base.stride_w = stride;
  base.pad_h = pad;
  base.pad_w = pad;

  if (!st_conv2d_output_hw(h, w, k, k, &base, &out_h, &out_w)) {
    return 1;
  }

  FloatTensor *input = make4d(n, c_in, h, w);
  FloatTensor *weight = make4d(c_out, c_in, k, k);
  FloatTensor *ref = make4d(n, c_out, out_h, out_w);
  FloatTensor *out = make4d(n, c_out, out_h, out_w);
  if (!input || !weight || !ref || !out) {
    goto cleanup;
  }

  fill_rand(input, 1u);
  fill_rand(weight, 2u);

  StConv2dParams ref_params = base;
  ref_params.backend = ST_CONV_BACKEND_GEMM;
  if (!st_conv2d_nchw(input, weight, NULL, &ref_params, ref)) {
    goto cleanup;
  }

  print_header();

  if (should_run_variant(requested_variant, "mps_zero_copy_sync")) {
    st_backend_reset_counters();
    st_backend_set_conv_mps_async(true);
    if (run_variant(case_name, "mps_zero_copy_sync", &base, input, weight, ref,
                    out, true, true) != 0) {
      goto cleanup;
    }
  }

  if (should_run_variant(requested_variant, "mps_true_async_boundary")) {
    st_backend_reset_counters();
    st_backend_set_conv_mps_async(true);
    if (run_variant(case_name, "mps_true_async_boundary", &base, input, weight,
                    ref, out, false, true) != 0) {
      goto cleanup;
    }
  }

  rc = 0;

cleanup:
  st_backend_set_conv_mps_async(false);
  st_destroy(out);
  st_destroy(ref);
  st_destroy(weight);
  st_destroy(input);
  return rc;
}
