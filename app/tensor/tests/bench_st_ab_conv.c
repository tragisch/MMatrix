/* bench_st_ab_conv.c — explicit A/B comparison: forced GEMM vs forced MPS
 *
 * Runs identical conv shapes back-to-back with ST_CONV_BACKEND_GEMM and
 * ST_CONV_BACKEND_MPS so there is no threshold heuristic / fallback
 * contamination.  Each backend is warmed up independently.  If the MPS call
 * returns false (kernel unsupported for a given shape), the row is marked
 * "mps_unsupported" instead of silently falling back to CPU.
 *
 * CSV columns match bench_st_canonical.c for easy comparison.
 */
#include "st.h"
#include "st_backend.h"
#include "st_conv.h"
#include "log.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- helpers ------------------------------------------------------------- */

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

static FloatTensor *make4d_cpu(size_t n, size_t c, size_t h, size_t w) {
  size_t shape[4] = {n, c, h, w};
  size_t numel = n * c * h * w;
  float *data = (float *)calloc(numel, sizeof(float));
  if (!data) return NULL;
  return st_create_with_data(4, shape, data, numel, true);
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

typedef struct { long mps_hit, mps_miss, fallback_gemm, fallback_ref; }
  BenchCountersDelta;

static BenchCountersDelta counters_delta(StBackendCounters before,
                                         StBackendCounters after) {
  return (BenchCountersDelta){
    .mps_hit      = after.mps_hit      - before.mps_hit,
    .mps_miss     = after.mps_miss     - before.mps_miss,
    .fallback_gemm = after.fallback_gemm - before.fallback_gemm,
    .fallback_ref  = after.fallback_ref  - before.fallback_ref,
  };
}

static void print_header(void) {
  printf("suite,case_name,variant,requested_backend,observed_backend,"
         "warmup,iters,ms_per_iter,mps_hit,mps_miss,"
         "fallback_gemm,fallback_ref,max_abs_diff,out_buf_type,readbytes_delta,fastpath_delta,"
         "fp_exec_nil_delta,fp_missing_feed_delta,fp_preout_nil_delta,fp_cmd_buf_nil_delta,fp_encode_exc_delta\n");
}

static void print_row(const char *case_name, const char *variant,
                      const char *req_backend, const char *obs_backend,
                      const char *out_buf_type, long readbytes_delta,
                      long fastpath_delta,
                      long fp_exec_nil_delta,
                      long fp_missing_feed_delta,
                      long fp_preout_nil_delta,
                      long fp_cmd_buf_nil_delta,
                      long fp_encode_exc_delta,
                      size_t warmup, size_t iters, double ms,
                      BenchCountersDelta d, float diff) {
  printf("conv_ab,%s,%s,%s,%s,%zu,%zu,%.6f,%ld,%ld,%ld,%ld,",
         case_name, variant, req_backend, obs_backend,
         warmup, iters, ms, d.mps_hit, d.mps_miss,
         d.fallback_gemm, d.fallback_ref);
  if (diff >= 0.0f) printf("%.8f", (double)diff);
  else              printf("na");
  printf(",%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld\n",
         out_buf_type, readbytes_delta, fastpath_delta,
         fp_exec_nil_delta, fp_missing_feed_delta, fp_preout_nil_delta,
         fp_cmd_buf_nil_delta, fp_encode_exc_delta);
}

static int run_mps_variant(const char *case_name, const char *variant,
                           const StConv2dParams *mps_params,
                           const FloatTensor *input,
                           const FloatTensor *weight,
                           const FloatTensor *out_ref,
                           FloatTensor *out,
                           size_t warmup, size_t iters,
                           bool sync_each_iter,
                           bool sync_at_end) {
  const bool out_is_metal =
      (out->buf && st_buffer_metal_handle(out->buf) != NULL);
  const char *out_buf_type = out_is_metal ? "metal" : "cpu";

  StBackendCounters probe_before = st_backend_get_counters();
  if (!st_conv2d_nchw(input, weight, NULL, mps_params, out)) {
    printf("conv_ab,%s,%s,mps,mps_hard_fail,0,0,na,0,0,0,0,na,%s,na,na,na,na,na,na,na\n",
           case_name, variant, out_buf_type);
    return 0;
  }
  if (sync_each_iter) st_tensor_sync(out);
  StBackendCounters probe_after = st_backend_get_counters();
  BenchCountersDelta probe_delta = counters_delta(probe_before, probe_after);
  bool probe_pure_mps = (probe_delta.mps_hit > 0 &&
                         probe_delta.mps_miss == 0 &&
                         probe_delta.fallback_gemm == 0 &&
                         probe_delta.fallback_ref == 0);
  if (!probe_pure_mps) {
    printf("conv_ab,%s,%s,mps,%s,0,0,na,%ld,%ld,%ld,%ld,na,%s,na,na,na,na,na,na,na\n",
           case_name, variant, st_conv2d_last_backend(),
           probe_delta.mps_hit, probe_delta.mps_miss,
           probe_delta.fallback_gemm, probe_delta.fallback_ref,
           out_buf_type);
    return 0;
  }

  for (size_t i = 1; i < warmup; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, mps_params, out)) return 1;
    if (sync_each_iter) st_tensor_sync(out);
  }

  StBackendCounters c0 = st_backend_get_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < iters; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, mps_params, out)) return 1;
    if (sync_each_iter) st_tensor_sync(out);
  }
  if (sync_at_end) st_tensor_sync(out);
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

  bool timed_pure_mps = (d.mps_hit > 0 &&
                         d.mps_miss == 0 &&
                         d.fallback_gemm == 0 &&
                         d.fallback_ref == 0);
  const char *mps_obs = st_conv2d_last_backend();
  if (!timed_pure_mps) {
    fprintf(stderr, "WARNING: %s %s loop fell back (%s) — row not a valid MPS comparison\n",
            case_name, variant, mps_obs);
  }

  print_row(case_name, variant, "mps", mps_obs,
            out_buf_type, readbytes_delta, fastpath_delta,
            fp_exec_nil_delta, fp_missing_feed_delta,
            fp_preout_nil_delta, fp_cmd_buf_nil_delta,
            fp_encode_exc_delta,
            warmup, iters, elapsed_ms(t0, t1, iters),
            d, timed_pure_mps ? max_abs_diff(out_ref, out) : -1.0f);
  return 0;
}

/* ---- single A/B case ----------------------------------------------------- */

static int bench_case(const char *case_name,
                      size_t n, size_t c_in, size_t c_out,
                      size_t h, size_t w, size_t k,
                      size_t stride, size_t pad,
                      size_t warmup, size_t iters) {
  int rc = 1;
  size_t out_h = 0, out_w = 0;

  StConv2dParams base = st_conv2d_default_params();
  base.stride_h = stride; base.stride_w = stride;
  base.pad_h    = pad;    base.pad_w    = pad;

  if (!st_conv2d_output_hw(h, w, k, k, &base, &out_h, &out_w)) return 1;

  FloatTensor *input  = make4d(n, c_in, h, w);
  FloatTensor *weight = make4d(c_out, c_in, k, k);
  FloatTensor *out_cpu = make4d(n, c_out, out_h, out_w);
  FloatTensor *out_mps = make4d(n, c_out, out_h, out_w);
  FloatTensor *out_mps_cpu = make4d_cpu(n, c_out, out_h, out_w);
  if (!input || !weight || !out_cpu || !out_mps || !out_mps_cpu) goto cleanup;

  const bool out_mps_is_metal =
      (out_mps->buf && st_buffer_metal_handle(out_mps->buf) != NULL);
  const char *out_mps_buf_type = out_mps_is_metal ? "metal" : "cpu";

  fill_rand(input,  1u);
  fill_rand(weight, 2u);

  StConv2dParams cpu_params = base;
  cpu_params.backend = ST_CONV_BACKEND_GEMM;

  StConv2dParams mps_params = base;
  mps_params.backend = ST_CONV_BACKEND_MPS;

  /* ---- A: GEMM warmup + timed ------------------------------------------ */
  for (size_t i = 0; i < warmup; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, &cpu_params, out_cpu)) goto cleanup;
  }

  StBackendCounters c0 = st_backend_get_counters();
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < iters; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, &cpu_params, out_cpu)) goto cleanup;
  }
  st_tensor_sync(out_cpu);
  uint64_t t1 = now_ns();
  StBackendCounters c1 = st_backend_get_counters();
  const long cpu_readbytes_delta = c1.conv_readbytes - c0.conv_readbytes;
  const long cpu_fastpath_delta = c1.conv_fastpath_hit - c0.conv_fastpath_hit;
    const long cpu_fp_exec_nil_delta =
      c1.conv_fastpath_executable_nil - c0.conv_fastpath_executable_nil;
    const long cpu_fp_missing_feed_delta =
      c1.conv_fastpath_missing_feed - c0.conv_fastpath_missing_feed;
    const long cpu_fp_preout_nil_delta =
      c1.conv_fastpath_preout_nil - c0.conv_fastpath_preout_nil;
    const long cpu_fp_cmd_buf_nil_delta =
      c1.conv_fastpath_cmd_buf_nil - c0.conv_fastpath_cmd_buf_nil;
    const long cpu_fp_encode_exc_delta =
      c1.conv_fastpath_encode_exception - c0.conv_fastpath_encode_exception;

  print_row(case_name, "gemm", "gemm", st_conv2d_last_backend(),
            "cpu", cpu_readbytes_delta, cpu_fastpath_delta,
            cpu_fp_exec_nil_delta, cpu_fp_missing_feed_delta,
            cpu_fp_preout_nil_delta, cpu_fp_cmd_buf_nil_delta,
            cpu_fp_encode_exc_delta,
            warmup, iters, elapsed_ms(t0, t1, iters),
            counters_delta(c0, c1), -1.0f);

  (void)out_mps_buf_type;

  /* ---- B: MPS split modes ---------------------------------------------- */
  st_backend_set_conv_mps_async(true);

  if (run_mps_variant(case_name, "mps_zero_copy_sync", &mps_params,
                      input, weight, out_cpu, out_mps,
                      warmup, iters, true, true) != 0) {
    goto cleanup;
  }

  if (run_mps_variant(case_name, "mps_true_async_boundary", &mps_params,
                      input, weight, out_cpu, out_mps,
                      warmup, iters, false, true) != 0) {
    goto cleanup;
  }

  if (run_mps_variant(case_name, "mps_cpu_materialized", &mps_params,
                      input, weight, out_cpu, out_mps_cpu,
                      warmup, iters, false, false) != 0) {
    goto cleanup;
  }

  st_backend_set_conv_mps_async(false);

  rc = 0;
cleanup:
  st_backend_set_conv_mps_async(false);
  st_destroy(out_mps_cpu);
  st_destroy(out_mps);
  st_destroy(out_cpu);
  st_destroy(weight);
  st_destroy(input);
  return rc;
}

/* ---- main ---------------------------------------------------------------- */

int main(void) {
  log_set_level(LOG_WARN);
  print_header();

  /*  name          n   c_in c_out   h    w   k  str pad  wu  it  */
  bench_case("conv_small",  1,   8,   8,  16,  16, 3,  1,  1,  3, 30);
  bench_case("conv_medium", 4,  32,  64,  56,  56, 3,  1,  1,  3, 10);
  bench_case("conv_large",  8,  64, 128, 112, 112, 3,  1,  1,  2,  5);
  /* ResNet-like blocks */
  bench_case("resnet_s1",   1,  64,  64,  56,  56, 3,  1,  1,  3,  8);
  bench_case("resnet_s2",   1, 128, 256,  28,  28, 3,  1,  1,  3,  8);
  bench_case("resnet_s3",   1, 256, 512,  14,  14, 3,  1,  1,  3, 10);
  /* 1×1 convolutions */
  bench_case("pw_medium",   4,  64, 128,  56,  56, 1,  1,  0,  3, 10);
  bench_case("pw_large",    4, 128, 256,  28,  28, 1,  1,  0,  3, 10);

  return 0;
}
