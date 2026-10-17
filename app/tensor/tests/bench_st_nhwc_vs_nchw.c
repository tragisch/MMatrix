/*
 * bench_st_nhwc_vs_nchw.c
 *
 * Compares standalone MPSGraph conv2d with NCHW (current) vs. NHWC (experiment).
 * Layout switch is done via st_backend_set_conv_mps_nhwc(); no host-side copies.
 * The NHWC path inserts two GPU-level transposes inside the MPSGraph.
 *
 * Output (CSV):
 *   suite,case_name,n,c_in,c_out,h,w,k,nchw_ms,nhwc_ms,nhwc_vs_nchw_pct
 *
 * nhwc_vs_nchw_pct > 0  → NHWC is slower
 * nhwc_vs_nchw_pct < 0  → NHWC is faster
 */

#include "log.h"
#include "st.h"
#include "st_backend.h"
#include "st_conv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
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
  if (!a || !b || a->numel != b->numel) return FLT_MAX;
  for (size_t i = 0; i < a->numel; ++i) {
    float d = fabsf(a->values[i] - b->values[i]);
    if (d > max_abs) max_abs = d;
  }
  return max_abs;
}

/* ------------------------------------------------------------------ */

typedef struct Case {
  const char *name;
  size_t N, C_in, C_out, H, W, K, pad, stride;
} Case;

static const int WARMUP = 10;
static const int ITERS  = 50;

static void run_case(const Case *c) {
  const size_t K      = c->K;
  const size_t pad    = c->pad;
  const size_t stride = c->stride;
  const size_t H_out  = (c->H + 2 * pad - K) / stride + 1;
  const size_t W_out  = (c->W + 2 * pad - K) / stride + 1;

  FloatTensor *in       = make4d(c->N, c->C_in,  c->H,   c->W);
  FloatTensor *w        = make4d(c->C_out, c->C_in, K,    K);
  FloatTensor *out_nchw = make4d(c->N, c->C_out, H_out, W_out);
  FloatTensor *out_nhwc = make4d(c->N, c->C_out, H_out, W_out);
  if (!in || !w || !out_nchw || !out_nhwc) {
    fprintf(stderr, "SKIP %s: allocation failed\n", c->name);
    st_destroy(in); st_destroy(w); st_destroy(out_nchw); st_destroy(out_nhwc);
    return;
  }

  fill_rand(in, 42u);
  fill_rand(w,  7u);

  StConv2dParams params = {
    .stride_h = stride, .stride_w = stride,
    .pad_h    = pad,    .pad_w    = pad,
    .dilation_h = 1,    .dilation_w = 1,
    .backend  = ST_CONV_BACKEND_MPS,
  };

  /* ---------- NCHW path ---------- */
  st_backend_set_conv_mps_nhwc(false);

  for (int i = 0; i < WARMUP; ++i) {
    st_conv2d_nchw(in, w, NULL, &params, out_nchw);
  }

  double nchw_start = now_ms();
  for (int i = 0; i < ITERS; ++i) {
    st_conv2d_nchw(in, w, NULL, &params, out_nchw);
  }
  double nchw_ms = (now_ms() - nchw_start) / ITERS;

  /* ---------- NHWC path ---------- */
  st_backend_set_conv_mps_nhwc(true);

  for (int i = 0; i < WARMUP; ++i) {
    st_conv2d_nchw(in, w, NULL, &params, out_nhwc);
  }

  double nhwc_start = now_ms();
  for (int i = 0; i < ITERS; ++i) {
    st_conv2d_nchw(in, w, NULL, &params, out_nhwc);
  }
  double nhwc_ms = (now_ms() - nhwc_start) / ITERS;

  /* Reset to NCHW */
  st_backend_set_conv_mps_nhwc(false);

  const float max_abs = max_abs_diff(out_nchw, out_nhwc);
  if (!(max_abs <= 1e-3f)) {
    fprintf(stderr,
            "SKIP %s: NHWC output mismatch (max_abs=%.8f)\n",
            c->name, (double)max_abs);
    st_destroy(in);
    st_destroy(w);
    st_destroy(out_nchw);
    st_destroy(out_nhwc);
    return;
  }

  double diff_pct = (nhwc_ms - nchw_ms) / nchw_ms * 100.0;

  printf("nhwc_bench,%s,%zu,%zu,%zu,%zu,%zu,%zu,%.6f,%.6f,%.2f\n",
         c->name,
         c->N, c->C_in, c->C_out, c->H, c->W, K,
         nchw_ms, nhwc_ms, diff_pct);

  st_destroy(in);
  st_destroy(w);
  st_destroy(out_nchw);
  st_destroy(out_nhwc);
}

int main(void) {
  log_set_level(LOG_WARN);

  printf("suite,case_name,n,c_in,c_out,h,w,k,"
         "nchw_ms,nhwc_ms,nhwc_vs_nchw_pct\n");

  /* nhwc_vs_nchw_pct > 0: NHWC slower;  < 0: NHWC faster */

  static const Case cases[] = {
    /* name,          N,  C_in,  C_out,  H,   W,   K, pad, stride */
    {"conv_small",    1,   8,     8,    16,  16,   3,  1,   1},
    {"conv_medium",   4,  32,    64,    56,  56,   3,  1,   1},
    {"resnet_s1",     1,  64,    64,    56,  56,   3,  1,   1},
    {"pw_medium",     4,  64,   128,    56,  56,   1,  0,   1},
    {"conv_large",    1, 128,   256,    28,  28,   3,  1,   1},
    {"pw_large",      1, 256,   512,    14,  14,   1,  0,   1},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    run_case(&cases[i]);
  }

  return 0;
}
