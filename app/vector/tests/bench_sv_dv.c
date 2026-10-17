/*
 * Benchmark: sv / dv wrapper vs. Accelerate direct calls
 *
 * Measures overhead of our FloatVector / DoubleVector API compared to calling
 * cblas_sdot / cblas_ddot / cblas_saxpy / cblas_snrm2 directly via Accelerate.
 *
 * Build & run:
 *   bazel run --define MATRIX_BACKEND=accelerate //app/vector:bench_sv_dv
 */

#include "dv.h"
#include "sv.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define HAVE_BLAS 1
#elif defined(USE_OPENBLAS)
#include <cblas.h>
#define HAVE_BLAS 1
#else
#define HAVE_BLAS 0
#endif

/* ---- timing ------------------------------------------------------------ */

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* ---- helpers ----------------------------------------------------------- */

static void fill_rand_f(float *buf, size_t n, uint32_t seed) {
  for (size_t i = 0; i < n; i++) {
    seed = seed * 1664525u + 1013904223u;
    buf[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

static void fill_rand_d(double *buf, size_t n, uint32_t seed) {
  for (size_t i = 0; i < n; i++) {
    seed = seed * 1664525u + 1013904223u;
    buf[i] = ((double)(seed >> 8) / 16777216.0) - 0.5;
  }
}

/* ---- benchmark runner -------------------------------------------------- */

#define WARMUP 5
#define ITERS  200

typedef double (*BenchFn)(void *ctx);

static double run_bench(BenchFn fn, void *ctx, const char *label,
                        size_t n_elems) {
  /* warmup */
  for (int i = 0; i < WARMUP; i++) fn(ctx);

  uint64_t t0 = now_ns();
  volatile double sink = 0.0;
  for (int i = 0; i < ITERS; i++) sink += fn(ctx);
  uint64_t t1 = now_ns();
  (void)sink;

  double ms_per_iter = (double)(t1 - t0) / 1e6 / ITERS;
  double gb_s = (double)n_elems * sizeof(float) / 1e9 / (ms_per_iter / 1000.0);
  printf("  %-36s  %7.3f ms/iter  %6.2f GB/s\n", label, ms_per_iter, gb_s);
  return ms_per_iter;
}

/* ======================================================================== */
/* sv_dot  ----------------------------------------------------------------- */

typedef struct { FloatVector *a; FloatVector *b; } SvPair;
typedef struct { float *a; float *b; int n; } RawPair;

static double bench_sv_dot(void *ctx) {
  SvPair *p = ctx;
  return (double)sv_dot(p->a, p->b);
}

#if HAVE_BLAS
static double bench_raw_sdot(void *ctx) {
  RawPair *p = ctx;
  return (double)cblas_sdot(p->n, p->a, 1, p->b, 1);
}
#endif

/* sv_axpy ----------------------------------------------------------------- */

typedef struct { FloatVector *dst; FloatVector *src; } SvAxpyCtx;
typedef struct { float *dst; float *src; int n; } RawAxpyCtx;

static double bench_sv_axpy(void *ctx) {
  SvAxpyCtx *p = ctx;
  sv_axpy(p->dst, 2.0f, p->src);
  return 0.0;
}

#if HAVE_BLAS
static double bench_raw_saxpy(void *ctx) {
  RawAxpyCtx *p = ctx;
  cblas_saxpy(p->n, 2.0f, p->src, 1, p->dst, 1);
  return 0.0;
}
#endif

/* sv_norm_l2 -------------------------------------------------------------- */

typedef struct { FloatVector *v; } SvNormCtx;

static double bench_sv_norm(void *ctx) {
  SvNormCtx *p = ctx;
  return (double)sv_norm_l2(p->v);
}

#if HAVE_BLAS
typedef struct { float *v; int n; } RawNormCtx;
static double bench_raw_snrm2(void *ctx) {
  RawNormCtx *p = ctx;
  return (double)cblas_snrm2(p->n, p->v, 1);
}
#endif

/* dv_dot ------------------------------------------------------------------ */

typedef struct { DoubleVector *a; DoubleVector *b; } DvPair;

static double bench_dv_dot(void *ctx) {
  DvPair *p = ctx;
  return dv_dot(p->a, p->b);
}

#if HAVE_BLAS
typedef struct { double *a; double *b; int n; } RawDPair;
static double bench_raw_ddot(void *ctx) {
  RawDPair *p = ctx;
  return cblas_ddot(p->n, p->a, 1, p->b, 1);
}
#endif

/* ======================================================================== */

int main(void) {
  const size_t sizes[] = {256, 4096, 65536, 1048576};
  const int    nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

  printf("bench_sv_dv  —  wrapper overhead vs. Accelerate direct\n");
  printf("WARMUP=%d  ITERS=%d\n\n", WARMUP, ITERS);

#if !HAVE_BLAS
  printf("WARNING: no BLAS backend — direct calls not available.\n"
         "Build with --define MATRIX_BACKEND=accelerate\n\n");
#endif

  for (int si = 0; si < nsizes; si++) {
    size_t n = sizes[si];
    printf("──── n = %zu ────────────────────────────────────────────\n", n);

    /* allocate raw buffers */
    float  *fa = malloc(n * sizeof(float));
    float  *fb = malloc(n * sizeof(float));
    double *da = malloc(n * sizeof(double));
    double *db = malloc(n * sizeof(double));
    fill_rand_f(fa, n, 0xDEAD1234u);
    fill_rand_f(fb, n, 0xBEEF5678u);
    fill_rand_d(da, n, 0xCAFEBABEu);
    fill_rand_d(db, n, 0xF00DC0DEu);

    /* wrap in our types (we own the data — use create_with_values to copy) */
    FloatVector  *sva = sv_create_with_values(n, fa);
    FloatVector  *svb = sv_create_with_values(n, fb);
    DoubleVector *dva = dv_create_with_values(n, da);
    DoubleVector *dvb = dv_create_with_values(n, db);

    /* --- dot --- */
    SvPair sv_pair = {sva, svb};
    run_bench(bench_sv_dot, &sv_pair, "sv_dot (wrapper)", n * 2);
#if HAVE_BLAS
    RawPair raw_pair = {fa, fb, (int)n};
    run_bench(bench_raw_sdot, &raw_pair, "cblas_sdot (direct)", n * 2);
#endif

    /* --- axpy --- */
    SvAxpyCtx sv_axpy_ctx = {sva, svb};
    run_bench(bench_sv_axpy, &sv_axpy_ctx, "sv_axpy (wrapper)", n * 3);
#if HAVE_BLAS
    float  *axpy_dst = malloc(n * sizeof(float));
    memcpy(axpy_dst, fa, n * sizeof(float));
    RawAxpyCtx raw_axpy = {axpy_dst, fb, (int)n};
    run_bench(bench_raw_saxpy, &raw_axpy, "cblas_saxpy (direct)", n * 3);
    free(axpy_dst);
#endif

    /* --- norm --- */
    SvNormCtx sv_norm_ctx = {sva};
    run_bench(bench_sv_norm, &sv_norm_ctx, "sv_norm_l2 (wrapper)", n);
#if HAVE_BLAS
    RawNormCtx raw_norm = {fa, (int)n};
    run_bench(bench_raw_snrm2, &raw_norm, "cblas_snrm2 (direct)", n);
#endif

    /* --- dv_dot --- */
    DvPair dv_pair = {dva, dvb};
    run_bench(bench_dv_dot, &dv_pair, "dv_dot (wrapper)", n * 2);
#if HAVE_BLAS
    RawDPair raw_dpair = {da, db, (int)n};
    run_bench(bench_raw_ddot, &raw_dpair, "cblas_ddot (direct)", n * 2);
#endif

    sv_destroy(sva);
    sv_destroy(svb);
    dv_destroy(dva);
    dv_destroy(dvb);
    free(fa); free(fb); free(da); free(db);
    printf("\n");
  }
  return 0;
}
