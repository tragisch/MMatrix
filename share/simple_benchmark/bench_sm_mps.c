/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * bench_sm_mps — compare CPU BLAS-backed sm_gemm with explicit MPS one-shot
 * matrix multiplication. This intentionally measures the current CPU-boundary
 * API shape: host inputs, synchronous GPU execution, host output.
 */

#include "simple_bench.h"
#include "sm.h"
#include "sm_mps.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct BenchCase {
  size_t m;
  size_t k;
  size_t n;
  size_t warmup;
  size_t iters;
} BenchCase;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double elapsed_ms(uint64_t start, uint64_t end, size_t iters) {
  return (double)(end - start) / 1000000.0 / (double)iters;
}

static void fill_rand(float *values, size_t count, uint32_t seed) {
  for (size_t i = 0; i < count; ++i) {
    seed = seed * 1664525u + 1013904223u;
    values[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

static float max_abs_diff(const float *a, const float *b, size_t count) {
  float max_abs = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    float d = fabsf(a[i] - b[i]);
    if (d > max_abs) {
      max_abs = d;
    }
  }
  return max_abs;
}

static void zero(float *values, size_t count) {
  memset(values, 0, count * sizeof(float));
}

static bool copy_to_mps_matrix(SmMpsMatrix *matrix, const float *values,
                               size_t count) {
  float *contents = sm_mps_matrix_contents(matrix);
  if (!contents || !values) {
    return false;
  }
  memcpy(contents, values, count * sizeof(float));
  return true;
}

static bool copy_from_mps_matrix(const SmMpsMatrix *matrix, float *values,
                                 size_t count) {
  const float *contents = sm_mps_matrix_const_contents(matrix);
  if (!contents || !values) {
    return false;
  }
  memcpy(values, contents, count * sizeof(float));
  return true;
}

typedef struct CounterDelta {
  unsigned long long matrix_allocations;
  unsigned long long command_buffers_created;
  unsigned long long commits;
  unsigned long long waits;
  unsigned long long gemm_encodes;
  unsigned long long uploads;
  unsigned long long downloads;
  unsigned long long plan_allocations;
} CounterDelta;

static CounterDelta counter_delta(SmMpsCounters before, SmMpsCounters after) {
  return (CounterDelta){
      .matrix_allocations = after.matrix_allocations - before.matrix_allocations,
      .command_buffers_created =
          after.command_buffers_created - before.command_buffers_created,
      .commits = after.commits - before.commits,
      .waits = after.waits - before.waits,
      .gemm_encodes = after.gemm_encodes - before.gemm_encodes,
      .uploads = after.uploads - before.uploads,
      .downloads = after.downloads - before.downloads,
      .plan_allocations = after.plan_allocations - before.plan_allocations,
  };
}

static void print_counter_delta(const char *shape, const char *variant,
                                CounterDelta d) {
  fprintf(stderr,
          "# counters shape=%s variant=%s matrices=%llu cmd_buf=%llu commits=%llu"
          " waits=%llu encodes=%llu uploads=%llu downloads=%llu plans=%llu\n",
          shape, variant, d.matrix_allocations, d.command_buffers_created,
          d.commits, d.waits, d.gemm_encodes, d.uploads, d.downloads,
          d.plan_allocations);
}

static bool run_resident_gemm(const BenchCase *tc, const float *a_values,
                              const float *b_values, float *out_values,
                              double *ms_out) {
  SmMpsMatrix *A = sm_mps_matrix_create(tc->m, tc->k);
  SmMpsMatrix *B = sm_mps_matrix_create(tc->k, tc->n);
  SmMpsMatrix *C = sm_mps_matrix_create(tc->m, tc->n);
  if (!A || !B || !C) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    return false;
  }
  if (!sm_mps_matrix_upload(A, a_values) ||
      !sm_mps_matrix_upload(B, b_values)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    return false;
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    if (!sm_mps_matrix_gemm_ex(C, 1.0f, A, false, B, false, 0.0f)) {
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      return false;
    }
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    if (!sm_mps_matrix_gemm_ex(C, 1.0f, A, false, B, false, 0.0f)) {
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      return false;
    }
  }
  uint64_t end = now_ns();
  if (!sm_mps_matrix_download(C, out_values)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    return false;
  }

  *ms_out = elapsed_ms(start, end, tc->iters);
  sm_mps_matrix_destroy(A);
  sm_mps_matrix_destroy(B);
  sm_mps_matrix_destroy(C);
  return true;
}

static bool run_resident_async_batch(const BenchCase *tc, const float *a_values,
                                     const float *b_values, float *out_values,
                                     double *ms_out) {
  SmMpsMatrix *A = sm_mps_matrix_create(tc->m, tc->k);
  SmMpsMatrix *B = sm_mps_matrix_create(tc->k, tc->n);
  SmMpsMatrix *C = sm_mps_matrix_create(tc->m, tc->n);
  if (!A || !B || !C) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    return false;
  }
  if (!sm_mps_matrix_upload(A, a_values) ||
      !sm_mps_matrix_upload(B, b_values)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    return false;
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    SmMpsStream *warmup_stream = sm_mps_stream_create();
    if (!warmup_stream ||
        !sm_mps_matrix_gemm_async(warmup_stream, C, 1.0f, A, false, B, false,
                                  0.0f) ||
        !sm_mps_stream_wait(warmup_stream)) {
      sm_mps_stream_destroy(warmup_stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      return false;
    }
    sm_mps_stream_destroy(warmup_stream);
  }

  SmMpsStream *stream = sm_mps_stream_create();
  if (!stream) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    return false;
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    if (!sm_mps_matrix_gemm_async(stream, C, 1.0f, A, false, B, false, 0.0f)) {
      sm_mps_stream_destroy(stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      return false;
    }
  }
  if (!sm_mps_stream_wait(stream)) {
    sm_mps_stream_destroy(stream);
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    return false;
  }
  uint64_t end = now_ns();
  sm_mps_stream_destroy(stream);

  if (!sm_mps_matrix_download(C, out_values)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    return false;
  }

  *ms_out = elapsed_ms(start, end, tc->iters);
  sm_mps_matrix_destroy(A);
  sm_mps_matrix_destroy(B);
  sm_mps_matrix_destroy(C);
  return true;
}

static bool run_resident_async_plan(const BenchCase *tc, const float *a_values,
                                    const float *b_values, float *out_values,
                                    double *ms_out) {
  SmMpsMatrix *A = sm_mps_matrix_create(tc->m, tc->k);
  SmMpsMatrix *B = sm_mps_matrix_create(tc->k, tc->n);
  SmMpsMatrix *C = sm_mps_matrix_create(tc->m, tc->n);
  SmMpsGemmPlan *plan =
      sm_mps_gemm_plan_create(tc->m, tc->n, tc->k, false, false, 1.0f, 0.0f);
  if (!A || !B || !C || !plan) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }
  if (!sm_mps_matrix_upload(A, a_values) ||
      !sm_mps_matrix_upload(B, b_values)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    SmMpsStream *warmup_stream = sm_mps_stream_create();
    if (!warmup_stream ||
        !sm_mps_gemm_plan_encode(warmup_stream, plan, C, A, B) ||
        !sm_mps_stream_wait(warmup_stream)) {
      sm_mps_stream_destroy(warmup_stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_gemm_plan_destroy(plan);
      return false;
    }
    sm_mps_stream_destroy(warmup_stream);
  }

  SmMpsStream *stream = sm_mps_stream_create();
  if (!stream) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    if (!sm_mps_gemm_plan_encode(stream, plan, C, A, B)) {
      sm_mps_stream_destroy(stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_gemm_plan_destroy(plan);
      return false;
    }
  }
  if (!sm_mps_stream_wait(stream)) {
    sm_mps_stream_destroy(stream);
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }
  uint64_t end = now_ns();
  sm_mps_stream_destroy(stream);

  if (!sm_mps_matrix_download(C, out_values)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }

  *ms_out = elapsed_ms(start, end, tc->iters);
  sm_mps_matrix_destroy(A);
  sm_mps_matrix_destroy(B);
  sm_mps_matrix_destroy(C);
  sm_mps_gemm_plan_destroy(plan);
  return true;
}

static bool run_resident_direct_async_plan(const BenchCase *tc,
                                           const float *a_values,
                                           const float *b_values,
                                           float *out_values,
                                           double *ms_out) {
  SmMpsMatrix *A = sm_mps_matrix_create(tc->m, tc->k);
  SmMpsMatrix *B = sm_mps_matrix_create(tc->k, tc->n);
  SmMpsMatrix *C = sm_mps_matrix_create(tc->m, tc->n);
  SmMpsGemmPlan *plan =
      sm_mps_gemm_plan_create(tc->m, tc->n, tc->k, false, false, 1.0f, 0.0f);
  if (!A || !B || !C || !plan ||
      !copy_to_mps_matrix(A, a_values, tc->m * tc->k) ||
      !copy_to_mps_matrix(B, b_values, tc->k * tc->n)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    SmMpsStream *warmup_stream = sm_mps_stream_create();
    if (!warmup_stream ||
        !sm_mps_gemm_plan_encode(warmup_stream, plan, C, A, B) ||
        !sm_mps_stream_wait(warmup_stream)) {
      sm_mps_stream_destroy(warmup_stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_gemm_plan_destroy(plan);
      return false;
    }
    sm_mps_stream_destroy(warmup_stream);
  }

  SmMpsStream *stream = sm_mps_stream_create();
  if (!stream) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    if (!sm_mps_gemm_plan_encode(stream, plan, C, A, B)) {
      sm_mps_stream_destroy(stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_gemm_plan_destroy(plan);
      return false;
    }
  }
  if (!sm_mps_stream_wait(stream)) {
    sm_mps_stream_destroy(stream);
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }
  uint64_t end = now_ns();
  sm_mps_stream_destroy(stream);
  bool ok = copy_from_mps_matrix(C, out_values, tc->m * tc->n);

  *ms_out = elapsed_ms(start, end, tc->iters);
  sm_mps_matrix_destroy(A);
  sm_mps_matrix_destroy(B);
  sm_mps_matrix_destroy(C);
  sm_mps_gemm_plan_destroy(plan);
  return ok;
}

static bool run_resident_bias_relu_sync(const BenchCase *tc,
                                        const float *a_values,
                                        const float *b_values,
                                        const float *bias_values,
                                        float *out_values,
                                        double *ms_out) {
  SmMpsMatrix *A = sm_mps_matrix_create(tc->m, tc->k);
  SmMpsMatrix *B = sm_mps_matrix_create(tc->k, tc->n);
  SmMpsMatrix *C = sm_mps_matrix_create(tc->m, tc->n);
  SmMpsMatrix *Bias = sm_mps_matrix_create(1, tc->n);
  if (!A || !B || !C || !Bias ||
      !sm_mps_matrix_upload(A, a_values) ||
      !sm_mps_matrix_upload(B, b_values) ||
      !sm_mps_matrix_upload(Bias, bias_values)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    return false;
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    if (!sm_mps_matrix_gemm_bias_relu_ex(C, 1.0f, A, false, B, false,
                                         Bias, true)) {
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_matrix_destroy(Bias);
      return false;
    }
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    if (!sm_mps_matrix_gemm_bias_relu_ex(C, 1.0f, A, false, B, false,
                                         Bias, true)) {
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_matrix_destroy(Bias);
      return false;
    }
  }
  uint64_t end = now_ns();
  bool ok = sm_mps_matrix_download(C, out_values);

  *ms_out = elapsed_ms(start, end, tc->iters);
  sm_mps_matrix_destroy(A);
  sm_mps_matrix_destroy(B);
  sm_mps_matrix_destroy(C);
  sm_mps_matrix_destroy(Bias);
  return ok;
}

static bool run_resident_bias_relu_async_batch(const BenchCase *tc,
                                               const float *a_values,
                                               const float *b_values,
                                               const float *bias_values,
                                               float *out_values,
                                               double *ms_out) {
  SmMpsMatrix *A = sm_mps_matrix_create(tc->m, tc->k);
  SmMpsMatrix *B = sm_mps_matrix_create(tc->k, tc->n);
  SmMpsMatrix *C = sm_mps_matrix_create(tc->m, tc->n);
  SmMpsMatrix *Bias = sm_mps_matrix_create(1, tc->n);
  if (!A || !B || !C || !Bias ||
      !sm_mps_matrix_upload(A, a_values) ||
      !sm_mps_matrix_upload(B, b_values) ||
      !sm_mps_matrix_upload(Bias, bias_values)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    return false;
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    SmMpsStream *warmup_stream = sm_mps_stream_create();
    if (!warmup_stream ||
        !sm_mps_matrix_gemm_bias_relu_async(warmup_stream, C, 1.0f, A, false,
                                            B, false, Bias, true) ||
        !sm_mps_stream_wait(warmup_stream)) {
      sm_mps_stream_destroy(warmup_stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_matrix_destroy(Bias);
      return false;
    }
    sm_mps_stream_destroy(warmup_stream);
  }

  SmMpsStream *stream = sm_mps_stream_create();
  if (!stream) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    return false;
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    if (!sm_mps_matrix_gemm_bias_relu_async(stream, C, 1.0f, A, false,
                                            B, false, Bias, true)) {
      sm_mps_stream_destroy(stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_matrix_destroy(Bias);
      return false;
    }
  }
  if (!sm_mps_stream_wait(stream)) {
    sm_mps_stream_destroy(stream);
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    return false;
  }
  uint64_t end = now_ns();
  sm_mps_stream_destroy(stream);
  bool ok = sm_mps_matrix_download(C, out_values);

  *ms_out = elapsed_ms(start, end, tc->iters);
  sm_mps_matrix_destroy(A);
  sm_mps_matrix_destroy(B);
  sm_mps_matrix_destroy(C);
  sm_mps_matrix_destroy(Bias);
  return ok;
}

static bool run_resident_bias_relu_async_plan(const BenchCase *tc,
                                              const float *a_values,
                                              const float *b_values,
                                              const float *bias_values,
                                              float *out_values,
                                              double *ms_out) {
  SmMpsMatrix *A = sm_mps_matrix_create(tc->m, tc->k);
  SmMpsMatrix *B = sm_mps_matrix_create(tc->k, tc->n);
  SmMpsMatrix *C = sm_mps_matrix_create(tc->m, tc->n);
  SmMpsMatrix *Bias = sm_mps_matrix_create(1, tc->n);
  SmMpsGemmPlan *plan =
      sm_mps_gemm_plan_create(tc->m, tc->n, tc->k, false, false, 1.0f, 0.0f);
  if (!A || !B || !C || !Bias || !plan ||
      !sm_mps_matrix_upload(A, a_values) ||
      !sm_mps_matrix_upload(B, b_values) ||
      !sm_mps_matrix_upload(Bias, bias_values)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    SmMpsStream *warmup_stream = sm_mps_stream_create();
    if (!warmup_stream ||
        !sm_mps_gemm_plan_encode(warmup_stream, plan, C, A, B) ||
        !sm_mps_matrix_bias_relu_async(warmup_stream, C, Bias, true) ||
        !sm_mps_stream_wait(warmup_stream)) {
      sm_mps_stream_destroy(warmup_stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_matrix_destroy(Bias);
      sm_mps_gemm_plan_destroy(plan);
      return false;
    }
    sm_mps_stream_destroy(warmup_stream);
  }

  SmMpsStream *stream = sm_mps_stream_create();
  if (!stream) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    if (!sm_mps_gemm_plan_encode(stream, plan, C, A, B) ||
        !sm_mps_matrix_bias_relu_async(stream, C, Bias, true)) {
      sm_mps_stream_destroy(stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_matrix_destroy(Bias);
      sm_mps_gemm_plan_destroy(plan);
      return false;
    }
  }
  if (!sm_mps_stream_wait(stream)) {
    sm_mps_stream_destroy(stream);
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }
  uint64_t end = now_ns();
  sm_mps_stream_destroy(stream);
  bool ok = sm_mps_matrix_download(C, out_values);

  *ms_out = elapsed_ms(start, end, tc->iters);
  sm_mps_matrix_destroy(A);
  sm_mps_matrix_destroy(B);
  sm_mps_matrix_destroy(C);
  sm_mps_matrix_destroy(Bias);
  sm_mps_gemm_plan_destroy(plan);
  return ok;
}

static bool run_resident_bias_relu_direct_async_plan(const BenchCase *tc,
                                                     const float *a_values,
                                                     const float *b_values,
                                                     const float *bias_values,
                                                     float *out_values,
                                                     double *ms_out) {
  SmMpsMatrix *A = sm_mps_matrix_create(tc->m, tc->k);
  SmMpsMatrix *B = sm_mps_matrix_create(tc->k, tc->n);
  SmMpsMatrix *C = sm_mps_matrix_create(tc->m, tc->n);
  SmMpsMatrix *Bias = sm_mps_matrix_create(1, tc->n);
  SmMpsGemmPlan *plan =
      sm_mps_gemm_plan_create(tc->m, tc->n, tc->k, false, false, 1.0f, 0.0f);
  if (!A || !B || !C || !Bias || !plan ||
      !copy_to_mps_matrix(A, a_values, tc->m * tc->k) ||
      !copy_to_mps_matrix(B, b_values, tc->k * tc->n) ||
      !copy_to_mps_matrix(Bias, bias_values, tc->n)) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    SmMpsStream *warmup_stream = sm_mps_stream_create();
    if (!warmup_stream ||
        !sm_mps_gemm_plan_encode(warmup_stream, plan, C, A, B) ||
        !sm_mps_matrix_bias_relu_async(warmup_stream, C, Bias, true) ||
        !sm_mps_stream_wait(warmup_stream)) {
      sm_mps_stream_destroy(warmup_stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_matrix_destroy(Bias);
      sm_mps_gemm_plan_destroy(plan);
      return false;
    }
    sm_mps_stream_destroy(warmup_stream);
  }

  SmMpsStream *stream = sm_mps_stream_create();
  if (!stream) {
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    if (!sm_mps_gemm_plan_encode(stream, plan, C, A, B) ||
        !sm_mps_matrix_bias_relu_async(stream, C, Bias, true)) {
      sm_mps_stream_destroy(stream);
      sm_mps_matrix_destroy(A);
      sm_mps_matrix_destroy(B);
      sm_mps_matrix_destroy(C);
      sm_mps_matrix_destroy(Bias);
      sm_mps_gemm_plan_destroy(plan);
      return false;
    }
  }
  if (!sm_mps_stream_wait(stream)) {
    sm_mps_stream_destroy(stream);
    sm_mps_matrix_destroy(A);
    sm_mps_matrix_destroy(B);
    sm_mps_matrix_destroy(C);
    sm_mps_matrix_destroy(Bias);
    sm_mps_gemm_plan_destroy(plan);
    return false;
  }
  uint64_t end = now_ns();
  sm_mps_stream_destroy(stream);
  bool ok = copy_from_mps_matrix(C, out_values, tc->m * tc->n);

  *ms_out = elapsed_ms(start, end, tc->iters);
  sm_mps_matrix_destroy(A);
  sm_mps_matrix_destroy(B);
  sm_mps_matrix_destroy(C);
  sm_mps_matrix_destroy(Bias);
  sm_mps_gemm_plan_destroy(plan);
  return ok;
}

static int run_case(const BenchCase *tc) {
  const size_t a_count = tc->m * tc->k;
  const size_t b_count = tc->k * tc->n;
  const size_t c_count = tc->m * tc->n;

  FloatMatrix A = {.rows = tc->m, .cols = tc->k, .capacity = a_count};
  FloatMatrix B = {.rows = tc->k, .cols = tc->n, .capacity = b_count};
  FloatMatrix C = {.rows = tc->m, .cols = tc->n, .capacity = c_count};
  float *mps_out = (float *)calloc(c_count, sizeof(float));
  float *ref_out = (float *)calloc(c_count, sizeof(float));
  A.values = (float *)malloc(a_count * sizeof(float));
  B.values = (float *)malloc(b_count * sizeof(float));
  C.values = (float *)calloc(c_count, sizeof(float));

  if (!A.values || !B.values || !C.values || !mps_out || !ref_out) {
    fprintf(stderr, "allocation failed for %zux%zu x %zux%zu\n",
            tc->m, tc->k, tc->k, tc->n);
    free(A.values);
    free(B.values);
    free(C.values);
    free(mps_out);
    free(ref_out);
    return 1;
  }

  fill_rand(A.values, a_count, 1001u + (uint32_t)tc->m);
  fill_rand(B.values, b_count, 2001u + (uint32_t)tc->n);

  if (!sm_set_backend(SM_BACKEND_ACCELERATE)) {
    (void)sm_set_backend(SM_BACKEND_DEFAULT);
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    zero(C.values, c_count);
    if (!sm_gemm(&C, 1.0f, &A, SM_NO_TRANSPOSE, &B, SM_NO_TRANSPOSE, 0.0f)) {
      fprintf(stderr, "sm_gemm failed\n");
      return 1;
    }
  }

  uint64_t cpu_start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    zero(C.values, c_count);
    if (!sm_gemm(&C, 1.0f, &A, SM_NO_TRANSPOSE, &B, SM_NO_TRANSPOSE, 0.0f)) {
      fprintf(stderr, "sm_gemm failed\n");
      return 1;
    }
  }
  uint64_t cpu_end = now_ns();
  memcpy(ref_out, C.values, c_count * sizeof(float));

  if (!mps_get_shared_device()) {
    printf("sm_mps,%zux%zux%zu,mps_unavailable,na,na,na,na\n",
           tc->m, tc->k, tc->n);
    goto done;
  }
  char shape[64];
  snprintf(shape, sizeof(shape), "%zux%zux%zu", tc->m, tc->k, tc->n);

  sm_mps_reset_counters();
  for (size_t i = 0; i < tc->warmup; ++i) {
    zero(mps_out, c_count);
    if (!mps_matrix_multiply_ex(A.values, A.rows, A.cols, false,
                                B.values, B.rows, B.cols, false,
                                1.0f, 0.0f, mps_out, C.rows, C.cols)) {
      fprintf(stderr, "mps_matrix_multiply_ex failed\n");
      return 1;
    }
  }

  SmMpsCounters c_before = sm_mps_get_counters();
  uint64_t mps_start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    zero(mps_out, c_count);
    if (!mps_matrix_multiply_ex(A.values, A.rows, A.cols, false,
                                B.values, B.rows, B.cols, false,
                                1.0f, 0.0f, mps_out, C.rows, C.cols)) {
      fprintf(stderr, "mps_matrix_multiply_ex failed\n");
      return 1;
    }
  }
  uint64_t mps_end = now_ns();
  SmMpsCounters c_after = sm_mps_get_counters();

  double cpu_ms = elapsed_ms(cpu_start, cpu_end, tc->iters);
  double mps_ms = elapsed_ms(mps_start, mps_end, tc->iters);
  double speedup = cpu_ms / mps_ms;
  float diff = max_abs_diff(ref_out, mps_out, c_count);

  printf("sm_mps,%zux%zux%zu,oneshot,%zu,%zu,%.6f,%.6f,%.3f,%.6g\n",
         tc->m, tc->k, tc->n, tc->warmup, tc->iters,
         cpu_ms, mps_ms, speedup, (double)diff);
  fflush(stdout);
  print_counter_delta(shape, "oneshot", counter_delta(c_before, c_after));

  zero(mps_out, c_count);
  double resident_ms = 0.0;
  sm_mps_reset_counters();
  c_before = sm_mps_get_counters();
  if (!run_resident_gemm(tc, A.values, B.values, mps_out, &resident_ms)) {
    fprintf(stderr, "resident MPS GEMM failed\n");
    return 1;
  }
  c_after = sm_mps_get_counters();
  speedup = cpu_ms / resident_ms;
  diff = max_abs_diff(ref_out, mps_out, c_count);
  printf("sm_mps,%zux%zux%zu,resident_sync,%zu,%zu,%.6f,%.6f,%.3f,%.6g\n",
         tc->m, tc->k, tc->n, tc->warmup, tc->iters,
         cpu_ms, resident_ms, speedup, (double)diff);
  fflush(stdout);
  print_counter_delta(shape, "resident_sync", counter_delta(c_before, c_after));

  zero(mps_out, c_count);
  double async_batch_ms = 0.0;
  sm_mps_reset_counters();
  c_before = sm_mps_get_counters();
  if (!run_resident_async_batch(tc, A.values, B.values, mps_out,
                                &async_batch_ms)) {
    fprintf(stderr, "resident async batch MPS GEMM failed\n");
    return 1;
  }
  c_after = sm_mps_get_counters();
  speedup = cpu_ms / async_batch_ms;
  diff = max_abs_diff(ref_out, mps_out, c_count);
  printf("sm_mps,%zux%zux%zu,resident_async_batch,%zu,%zu,%.6f,%.6f,%.3f,%.6g\n",
         tc->m, tc->k, tc->n, tc->warmup, tc->iters,
         cpu_ms, async_batch_ms, speedup, (double)diff);
  fflush(stdout);
  print_counter_delta(shape, "resident_async_batch",
                      counter_delta(c_before, c_after));

  zero(mps_out, c_count);
  double async_plan_ms = 0.0;
  sm_mps_reset_counters();
  c_before = sm_mps_get_counters();
  if (!run_resident_async_plan(tc, A.values, B.values, mps_out,
                               &async_plan_ms)) {
    fprintf(stderr, "resident async plan MPS GEMM failed\n");
    return 1;
  }
  c_after = sm_mps_get_counters();
  speedup = cpu_ms / async_plan_ms;
  diff = max_abs_diff(ref_out, mps_out, c_count);
  printf("sm_mps,%zux%zux%zu,resident_async_plan,%zu,%zu,%.6f,%.6f,%.3f,%.6g\n",
         tc->m, tc->k, tc->n, tc->warmup, tc->iters,
         cpu_ms, async_plan_ms, speedup, (double)diff);
  fflush(stdout);
  print_counter_delta(shape, "resident_async_plan",
                      counter_delta(c_before, c_after));

  zero(mps_out, c_count);
  double direct_async_plan_ms = 0.0;
  sm_mps_reset_counters();
  c_before = sm_mps_get_counters();
  if (!run_resident_direct_async_plan(tc, A.values, B.values, mps_out,
                                      &direct_async_plan_ms)) {
    fprintf(stderr, "resident direct async plan MPS GEMM failed\n");
    return 1;
  }
  c_after = sm_mps_get_counters();
  speedup = cpu_ms / direct_async_plan_ms;
  diff = max_abs_diff(ref_out, mps_out, c_count);
  printf("sm_mps,%zux%zux%zu,resident_direct_async_plan,%zu,%zu,%.6f,%.6f,%.3f,%.6g\n",
         tc->m, tc->k, tc->n, tc->warmup, tc->iters,
         cpu_ms, direct_async_plan_ms, speedup, (double)diff);
  fflush(stdout);
  print_counter_delta(shape, "resident_direct_async_plan",
                      counter_delta(c_before, c_after));

done:
  free(A.values);
  free(B.values);
  free(C.values);
  free(mps_out);
  free(ref_out);
  return 0;
}

static int run_bias_relu_case(const BenchCase *tc) {
  const size_t a_count = tc->m * tc->k;
  const size_t b_count = tc->k * tc->n;
  const size_t c_count = tc->m * tc->n;

  FloatMatrix A = {.rows = tc->m, .cols = tc->k, .capacity = a_count};
  FloatMatrix B = {.rows = tc->k, .cols = tc->n, .capacity = b_count};
  FloatMatrix C = {.rows = tc->m, .cols = tc->n, .capacity = c_count};
  FloatMatrix Bias = {.rows = 1, .cols = tc->n, .capacity = tc->n};
  float *mps_out = (float *)calloc(c_count, sizeof(float));
  float *ref_out = (float *)calloc(c_count, sizeof(float));
  A.values = (float *)malloc(a_count * sizeof(float));
  B.values = (float *)malloc(b_count * sizeof(float));
  C.values = (float *)calloc(c_count, sizeof(float));
  Bias.values = (float *)malloc(tc->n * sizeof(float));

  if (!A.values || !B.values || !C.values || !Bias.values ||
      !mps_out || !ref_out) {
    fprintf(stderr, "allocation failed for bias_relu %zux%zu x %zux%zu\n",
            tc->m, tc->k, tc->k, tc->n);
    free(A.values);
    free(B.values);
    free(C.values);
    free(Bias.values);
    free(mps_out);
    free(ref_out);
    return 1;
  }

  fill_rand(A.values, a_count, 3001u + (uint32_t)tc->m);
  fill_rand(B.values, b_count, 4001u + (uint32_t)tc->n);
  fill_rand(Bias.values, tc->n, 5001u + (uint32_t)tc->k);

  if (!sm_set_backend(SM_BACKEND_ACCELERATE)) {
    (void)sm_set_backend(SM_BACKEND_DEFAULT);
  }

  for (size_t i = 0; i < tc->warmup; ++i) {
    zero(C.values, c_count);
    if (!sm_gemm_bias_relu(&C, &A, SM_NO_TRANSPOSE, &B, SM_NO_TRANSPOSE,
                           &Bias)) {
      fprintf(stderr, "sm_gemm_bias_relu failed\n");
      return 1;
    }
  }

  uint64_t cpu_start = now_ns();
  for (size_t i = 0; i < tc->iters; ++i) {
    zero(C.values, c_count);
    if (!sm_gemm_bias_relu(&C, &A, SM_NO_TRANSPOSE, &B, SM_NO_TRANSPOSE,
                           &Bias)) {
      fprintf(stderr, "sm_gemm_bias_relu failed\n");
      return 1;
    }
  }
  uint64_t cpu_end = now_ns();
  memcpy(ref_out, C.values, c_count * sizeof(float));

  if (!mps_get_shared_device()) {
    printf("sm_mps_bias_relu,%zux%zux%zu,mps_unavailable,na,na,na,na\n",
           tc->m, tc->k, tc->n);
    goto done;
  }

  char shape[64];
  snprintf(shape, sizeof(shape), "%zux%zux%zu", tc->m, tc->k, tc->n);
  const double cpu_ms = elapsed_ms(cpu_start, cpu_end, tc->iters);

  double mps_ms = 0.0;
  sm_mps_reset_counters();
  SmMpsCounters c_before = sm_mps_get_counters();
  if (!run_resident_bias_relu_sync(tc, A.values, B.values, Bias.values,
                                   mps_out, &mps_ms)) {
    fprintf(stderr, "resident bias_relu sync failed\n");
    return 1;
  }
  SmMpsCounters c_after = sm_mps_get_counters();
  float diff = max_abs_diff(ref_out, mps_out, c_count);
  printf("sm_mps_bias_relu,%zux%zux%zu,resident_sync,%zu,%zu,%.6f,%.6f,%.3f,%.6g\n",
         tc->m, tc->k, tc->n, tc->warmup, tc->iters,
         cpu_ms, mps_ms, cpu_ms / mps_ms, (double)diff);
  fflush(stdout);
  print_counter_delta(shape, "bias_relu_resident_sync",
                      counter_delta(c_before, c_after));

  zero(mps_out, c_count);
  sm_mps_reset_counters();
  c_before = sm_mps_get_counters();
  if (!run_resident_bias_relu_async_batch(tc, A.values, B.values, Bias.values,
                                          mps_out, &mps_ms)) {
    fprintf(stderr, "resident bias_relu async batch failed\n");
    return 1;
  }
  c_after = sm_mps_get_counters();
  diff = max_abs_diff(ref_out, mps_out, c_count);
  printf("sm_mps_bias_relu,%zux%zux%zu,resident_async_batch,%zu,%zu,%.6f,%.6f,%.3f,%.6g\n",
         tc->m, tc->k, tc->n, tc->warmup, tc->iters,
         cpu_ms, mps_ms, cpu_ms / mps_ms, (double)diff);
  fflush(stdout);
  print_counter_delta(shape, "bias_relu_resident_async_batch",
                      counter_delta(c_before, c_after));

  zero(mps_out, c_count);
  sm_mps_reset_counters();
  c_before = sm_mps_get_counters();
  if (!run_resident_bias_relu_async_plan(tc, A.values, B.values, Bias.values,
                                         mps_out, &mps_ms)) {
    fprintf(stderr, "resident bias_relu async plan failed\n");
    return 1;
  }
  c_after = sm_mps_get_counters();
  diff = max_abs_diff(ref_out, mps_out, c_count);
  printf("sm_mps_bias_relu,%zux%zux%zu,resident_async_plan,%zu,%zu,%.6f,%.6f,%.3f,%.6g\n",
         tc->m, tc->k, tc->n, tc->warmup, tc->iters,
         cpu_ms, mps_ms, cpu_ms / mps_ms, (double)diff);
  fflush(stdout);
  print_counter_delta(shape, "bias_relu_resident_async_plan",
                      counter_delta(c_before, c_after));

  zero(mps_out, c_count);
  sm_mps_reset_counters();
  c_before = sm_mps_get_counters();
  if (!run_resident_bias_relu_direct_async_plan(tc, A.values, B.values,
                                                Bias.values, mps_out,
                                                &mps_ms)) {
    fprintf(stderr, "resident bias_relu direct async plan failed\n");
    return 1;
  }
  c_after = sm_mps_get_counters();
  diff = max_abs_diff(ref_out, mps_out, c_count);
  printf("sm_mps_bias_relu,%zux%zux%zu,resident_direct_async_plan,%zu,%zu,%.6f,%.6f,%.3f,%.6g\n",
         tc->m, tc->k, tc->n, tc->warmup, tc->iters,
         cpu_ms, mps_ms, cpu_ms / mps_ms, (double)diff);
  fflush(stdout);
  print_counter_delta(shape, "bias_relu_resident_direct_async_plan",
                      counter_delta(c_before, c_after));

done:
  free(A.values);
  free(B.values);
  free(C.values);
  free(Bias.values);
  free(mps_out);
  free(ref_out);
  return 0;
}

int main(void) {
  const BenchCase cases[] = {
      {128, 128, 128, 5, 50},
      {256, 256, 256, 5, 30},
      {512, 512, 512, 3, 15},
      {1024, 1024, 1024, 2, 8},
      {2048, 2048, 2048, 1, 3},
  };

  print_environment_info();
  printf("suite,shape,variant,warmup,iters,cpu_ms_per_iter,mps_ms_per_iter,speedup_cpu_over_mps,max_abs_diff\n");
  fflush(stdout);
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    if (run_case(&cases[i]) != 0) {
      return 1;
    }
  }
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    if (run_bias_relu_case(&cases[i]) != 0) {
      return 1;
    }
  }
  return 0;
}
