/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * bench_st_async_release — A/B benchmark for st_buffer_release behavior.
 *
 * Compares:
 *   1) non-blocking release (default) via completion handler
 *   2) forced blocking release via MMATRIX_ST_BUFFER_RELEASE_BLOCKING=1
 *
 * Workload: async MPS Conv2D followed by immediate st_destroy(output)
 * in each iteration (release-path hot loop).
 */

#include "st.h"
#include "st_backend.h"
#include "st_conv.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double elapsed_ms(uint64_t t0, uint64_t t1, size_t iters) {
  return (double)(t1 - t0) / 1000000.0 / (double)iters;
}

static FloatTensor *make4d(size_t n, size_t c, size_t h, size_t w) {
  size_t s[4] = {n, c, h, w};
  return st_create(4, s);
}

static void fill_rand(FloatTensor *t, uint32_t seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1664525u + 1013904223u;
    t->values[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

static bool run_variant(const char *name, bool blocking_release, size_t warmup,
                        size_t iters) {
  if (blocking_release) {
    setenv("MMATRIX_ST_BUFFER_RELEASE_BLOCKING", "1", 1);
  } else {
    setenv("MMATRIX_ST_BUFFER_RELEASE_BLOCKING", "0", 1);
  }

  const size_t N = 4, Cin = 32, H = 56, W = 56;
  const size_t Cout = 64, K = 3;

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_MPS;
  p.stride_h = 1;
  p.stride_w = 1;
  p.pad_h = 1;
  p.pad_w = 1;

  size_t out_h = 0, out_w = 0;
  if (!st_conv2d_output_hw(H, W, K, K, &p, &out_h, &out_w)) {
    fprintf(stderr, "[SKIP] invalid conv shape\n");
    return false;
  }

  FloatTensor *input = make4d(N, Cin, H, W);
  FloatTensor *weight = make4d(Cout, Cin, K, K);
  if (!input || !weight) {
    st_destroy(input);
    st_destroy(weight);
    fprintf(stderr, "[OOM] setup failed\n");
    return false;
  }
  fill_rand(input, 1u);
  fill_rand(weight, 2u);

  const bool prev_async = st_backend_get_conv_mps_async();
  st_backend_set_conv_mps_async(true);

  FloatTensor *probe = make4d(N, Cout, out_h, out_w);
  bool probe_ok = probe && st_conv2d_nchw(input, weight, NULL, &p, probe);
  const char *last_backend = st_conv2d_last_backend();
  if (probe_ok) {
    st_tensor_sync(probe);
  }
  st_destroy(probe);

  if (!probe_ok || !last_backend || strcmp(last_backend, "mps") != 0) {
    fprintf(stderr, "[SKIP] MPS async path unavailable (backend=%s)\n",
            last_backend ? last_backend : "<null>");
    st_backend_set_conv_mps_async(prev_async);
    st_destroy(input);
    st_destroy(weight);
    return false;
  }

  for (size_t i = 0; i < warmup; ++i) {
    FloatTensor *out = make4d(N, Cout, out_h, out_w);
    if (!out) {
      break;
    }
    (void)st_conv2d_nchw(input, weight, NULL, &p, out);
    st_destroy(out);
  }

  st_buffer_pending_stats_reset();

  uint64_t t0 = now_ns();
  size_t ok_iters = 0;
  for (size_t i = 0; i < iters; ++i) {
    FloatTensor *out = make4d(N, Cout, out_h, out_w);
    if (!out) {
      break;
    }
    bool ok = st_conv2d_nchw(input, weight, NULL, &p, out);
    st_destroy(out);
    if (!ok) {
      break;
    }
    ++ok_iters;
  }
  uint64_t t1 = now_ns();

  st_backend_set_conv_mps_async(prev_async);
  st_destroy(input);
  st_destroy(weight);

  if (ok_iters == 0) {
    fprintf(stderr, "[FAIL] %s had zero successful iterations\n", name);
    return false;
  }

  const StBufferPendingStats stats = st_buffer_pending_stats_get();
  const double avg_depth =
      stats.samples > 0u ? (double)stats.total_depth / (double)stats.samples
                         : 0.0;

  printf("%s: %.3f ms/iter (%zu iters)\n", name, elapsed_ms(t0, t1, ok_iters),
         ok_iters);
  printf("  pending_depth: avg=%.2f max=%zu samples=%" PRIu64
         " enqueued=%" PRIu64 " evicted=%" PRIu64 "\n",
         avg_depth, stats.max_depth, stats.samples, stats.enqueued,
         stats.evicted);
  return true;
}

int main(void) {
  const size_t warmup = 20;
  const size_t iters = 200;

  printf("=== bench_st_async_release ===\n");
  printf("workload: async MPS Conv2D + immediate st_destroy(output)\n");
  printf("warmup=%zu iters=%zu\n\n", warmup, iters);

  bool ok_nb = run_variant("non_blocking_release", false, warmup, iters);
  bool ok_bl = run_variant("forced_blocking_release", true, warmup, iters);

  if (!ok_nb || !ok_bl) {
    printf("\nResult: SKIP/INCOMPLETE (see stderr)\n");
    return 0;
  }

  printf("\nResult: compare lines above (lower is better)\n");
  return 0;
}
