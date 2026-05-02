/*
 * bench_st_warmup.c — Measure first-run MPS latency with vs without warmup.
 *
 * For each shape, two scenarios are compared:
 *   COLD: fresh process-like run (cache cleared via process restart simulation
 *         is not feasible here, so we rely on measuring the 1st iteration
 *         vs subsequent iterations of the SAME op sequence to show the
 *         MPSGraph compilation cost amortisation effect).
 *   WARM: st_mps_warmup_shapes() called before measurement.
 *
 * Because MPSGraph caches within the same process, "cold" is measured
 * first (before any cache is populated), then "warm" is measured after
 * the explicit warmup call fills the cache.
 *
 * Usage:
 *   bazel run //app/tensor:bench_st_warmup
 *   MMATRIX_ST_MPS_WARMUP=1 bazel run //app/tensor:bench_st_warmup
 */

#include "st.h"
#include "st_backend.h"
#include "st_conv.h"
#include "st_pool.h"
#include "st_batchnorm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Timer ---- */
static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

/* ---- Helpers ---- */
static FloatTensor *make4d(size_t n, size_t c, size_t h, size_t w) {
  size_t s[4] = {n, c, h, w};
  return st_create(4, s);
}

static FloatTensor *make1d(size_t c) {
  size_t s[1] = {c};
  return st_create(1, s);
}

static void fill_rand(FloatTensor *t) {
  for (size_t i = 0; i < t->numel; ++i)
    t->values[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

/* ---- Measure single MaxPool2D run (first call timing) ---- */
static double time_first_maxpool2d(size_t n, size_t c, size_t h, size_t w,
                                    size_t kh, size_t kw,
                                    size_t sh, size_t sw) {
  const size_t oh = (h - kh) / sh + 1;
  const size_t ow = (w - kw) / sw + 1;
  FloatTensor *in  = make4d(n, c, h, w);
  FloatTensor *out = make4d(n, c, oh, ow);
  if (!in || !out) { st_destroy(in); st_destroy(out); return 0.0; }
  fill_rand(in);

  double t0 = now_ms();
  st_maxpool2d_nchw(in, kh, kw, sh, sw, 0, 0, out, NULL);
  double elapsed = now_ms() - t0;

  st_destroy(in); st_destroy(out);
  return elapsed;
}

/* ---- Measure single Conv2D run (first call timing) ---- */
static double time_first_conv2d(size_t n, size_t cin, size_t h, size_t w,
                                 size_t cout, size_t kh, size_t kw,
                                 size_t sh, size_t sw) {
  const size_t oh = (h - kh) / sh + 1;
  const size_t ow = (w - kw) / sw + 1;
  size_t sw4[4] = {n, cin, h, w};
  size_t ww4[4] = {cout, cin, kh, kw};
  size_t ow4[4] = {n, cout, oh, ow};

  FloatTensor *in     = st_create(4, sw4);
  FloatTensor *weight = st_create(4, ww4);
  FloatTensor *out    = st_create(4, ow4);
  if (!in || !weight || !out) {
    st_destroy(in); st_destroy(weight); st_destroy(out); return 0.0;
  }
  fill_rand(in); fill_rand(weight);

  StConv2dParams params = {
    .stride_h = sh, .stride_w = sw,
    .pad_h = 0, .pad_w = 0,
    .dilation_h = 1, .dilation_w = 1,
    .backend = ST_CONV_BACKEND_AUTO,
  };

  double t0 = now_ms();
  st_conv2d_nchw(in, weight, NULL, &params, out);
  double elapsed = now_ms() - t0;

  st_destroy(in); st_destroy(weight); st_destroy(out);
  return elapsed;
}

/* ---- Measure single BN run ---- */
static double time_first_batchnorm2d(size_t n, size_t c, size_t h, size_t w) {
  FloatTensor *in   = make4d(n, c, h, w);
  FloatTensor *out  = make4d(n, c, h, w);
  FloatTensor *mean = make1d(c);
  FloatTensor *var  = make1d(c);
  if (!in || !out || !mean || !var) {
    st_destroy(in); st_destroy(out); st_destroy(mean); st_destroy(var);
    return 0.0;
  }
  fill_rand(in);

  double t0 = now_ms();
  st_batchnorm2d_forward(in, NULL, NULL, 1e-5f, out, mean, var);
  double elapsed = now_ms() - t0;

  st_destroy(in); st_destroy(out); st_destroy(mean); st_destroy(var);
  return elapsed;
}

/* ------------------------------------------------------------------ */
int main(void) {
  srand(42);

  /* Print backend info */
  const StBackend *mps = st_backend_mps();
  printf("Backend: %s\n\n", mps ? "MPS available" : "MPS not available");

  /* ---- Shape used throughout: ResNet-50 early layer ---- */
  /* N=4, C=64→128, H=56, W=56, K=3×3, stride=1 */
  const size_t N = 4, Cin = 64, H = 56, W = 56;
  const size_t Cout = 128, K = 3, S = 1;

  printf("%-35s %10s %10s %10s\n",
         "Op (shape)", "COLD(ms)", "WARM(ms)", "Speedup");
  printf("%-35s %10s %10s %10s\n",
         "---", "---", "---", "---");

  /* --- COLD: measure first invocation (no prior warmup) --- */
  double cold_maxpool = time_first_maxpool2d(N, Cin, H, W, 3, 3, 2, 2);
  double cold_bn      = time_first_batchnorm2d(N, Cin, H, W);
  double cold_conv    = time_first_conv2d(N, Cin, H, W, Cout, K, K, S, S);

  /* --- WARMUP: populate the graph cache --- */
  StWarmupShape shapes[1] = {{
    .n = N, .c_in = Cin, .h = H, .w = W,
    .c_out = Cout, .kh = K, .kw = K,
    .sh = S, .sw = S, .ph = 0, .pw = 0,
  }};
  double t_wu0 = now_ms();
  st_mps_warmup_shapes(shapes, 1);
  double t_warmup = now_ms() - t_wu0;

  /* --- WARM: now measure with pre-compiled graphs --- */
  double warm_maxpool = time_first_maxpool2d(N, Cin, H, W, 3, 3, 2, 2);
  double warm_bn      = time_first_batchnorm2d(N, Cin, H, W);
  double warm_conv    = time_first_conv2d(N, Cin, H, W, Cout, K, K, S, S);

  /* --- Print results --- */
  printf("%-35s %10.3f %10.3f %10.2fx\n",
         "MaxPool2D N=4,C=64,56x56 K=3",
         cold_maxpool, warm_maxpool,
         cold_maxpool > 0 ? cold_maxpool / (warm_maxpool + 1e-9) : 0.0);

  printf("%-35s %10.3f %10.3f %10.2fx\n",
         "BatchNorm2D N=4,C=64,56x56",
         cold_bn, warm_bn,
         cold_bn > 0 ? cold_bn / (warm_bn + 1e-9) : 0.0);

  printf("%-35s %10.3f %10.3f %10.2fx\n",
         "Conv2D N=4,C=64->128,56x56 K=3",
         cold_conv, warm_conv,
         cold_conv > 0 ? cold_conv / (warm_conv + 1e-9) : 0.0);

  printf("\nWarmup cost: %.3f ms\n", t_warmup);
  printf("\nNote: COLD values include MPSGraph compilation on first call.\n");
  printf("      WARM values use the pre-compiled graph from the cache.\n");
  printf("      Run cold/warm in the same process; "
         "true cold requires a fresh process.\n");

  return 0;
}
