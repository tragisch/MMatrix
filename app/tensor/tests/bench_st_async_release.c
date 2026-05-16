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
  setenv("MMATRIX_ST_ASYNC_SYNC_EVERY", "0", 1);
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

static bool run_ring_variant(const char *name, bool blocking_release,
                             size_t warmup, size_t iters, size_t ring_size,
                             size_t sync_every, size_t auto_sync_every_env) {
  if (ring_size == 0u) {
    return false;
  }
  char auto_buf[32];
  snprintf(auto_buf, sizeof(auto_buf), "%zu", auto_sync_every_env);
  setenv("MMATRIX_ST_ASYNC_SYNC_EVERY", auto_buf, 1);

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
  fill_rand(input, 3u);
  fill_rand(weight, 4u);

  FloatTensor *outs[16] = {0};
  if (ring_size > 16u) {
    ring_size = 16u;
  }
  for (size_t i = 0; i < ring_size; ++i) {
    outs[i] = make4d(N, Cout, out_h, out_w);
    if (!outs[i]) {
      for (size_t j = 0; j < ring_size; ++j) {
        st_destroy(outs[j]);
      }
      st_destroy(input);
      st_destroy(weight);
      fprintf(stderr, "[OOM] output ring allocation failed\n");
      return false;
    }
  }

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
    for (size_t i = 0; i < ring_size; ++i) {
      st_destroy(outs[i]);
    }
    st_destroy(input);
    st_destroy(weight);
    return false;
  }

  for (size_t i = 0; i < warmup; ++i) {
    FloatTensor *out = outs[i % ring_size];
    (void)st_conv2d_nchw(input, weight, NULL, &p, out);
    if (sync_every > 0u && ((i + 1u) % sync_every) == 0u) {
      for (size_t j = 0; j < ring_size; ++j) {
        st_tensor_sync(outs[j]);
      }
    }
  }
  for (size_t i = 0; i < ring_size; ++i) {
    st_tensor_sync(outs[i]);
  }

  st_buffer_pending_stats_reset();

  uint64_t t0 = now_ns();
  size_t ok_iters = 0;
  for (size_t i = 0; i < iters; ++i) {
    FloatTensor *out = outs[i % ring_size];
    bool ok = st_conv2d_nchw(input, weight, NULL, &p, out);
    if (!ok) {
      break;
    }
    if (sync_every > 0u && ((i + 1u) % sync_every) == 0u) {
      for (size_t j = 0; j < ring_size; ++j) {
        st_tensor_sync(outs[j]);
      }
    }
    ++ok_iters;
  }
  for (size_t i = 0; i < ring_size; ++i) {
    st_tensor_sync(outs[i]);
  }
  uint64_t t1 = now_ns();

  st_backend_set_conv_mps_async(prev_async);
  for (size_t i = 0; i < ring_size; ++i) {
    st_destroy(outs[i]);
  }
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

      printf("%s: %.3f ms/iter (%zu iters, ring=%zu, sync_every=%zu, auto_sync=%zu)\n",
        name, elapsed_ms(t0, t1, ok_iters), ok_iters, ring_size, sync_every,
        auto_sync_every_env);
  printf("  pending_depth: avg=%.2f max=%zu samples=%" PRIu64
         " enqueued=%" PRIu64 " evicted=%" PRIu64 "\n",
         avg_depth, stats.max_depth, stats.samples, stats.enqueued,
         stats.evicted);
  return true;
}

int main(void) {
  const size_t warmup = 20;
  const size_t iters = 200;
  const size_t ring4 = 4;
  const size_t ring8 = 8;

  printf("=== bench_st_async_release ===\n");
  printf("workload: async MPS Conv2D + immediate st_destroy(output)\n");
  printf("warmup=%zu iters=%zu\n\n", warmup, iters);

  bool ok_nb = run_variant("non_blocking_release", false, warmup, iters);
  bool ok_bl = run_variant("forced_blocking_release", true, warmup, iters);

  printf("\nworkload: async MPS Conv2D + ring-reuse(no sync in-loop)\n");
  bool ok_ring_nb =
      run_ring_variant("ring4_non_blocking_release", false, warmup, iters,
               ring4, 0u, 0u);
  bool ok_ring_bl =
      run_ring_variant("ring4_forced_blocking_release", true, warmup, iters,
               ring4, 0u, 0u);

  printf("\nworkload: ring8 sync-cadence sweep\n");
  const size_t cadences[] = {0u, 4u, 8u, 16u};
  bool ok_sweep = true;
  for (size_t i = 0; i < sizeof(cadences) / sizeof(cadences[0]); ++i) {
    const size_t cadence = cadences[i];
    char name_nb[128];
    char name_bl[128];
    snprintf(name_nb, sizeof(name_nb), "ring8_non_blocking_sync%zu", cadence);
    snprintf(name_bl, sizeof(name_bl), "ring8_forced_blocking_sync%zu", cadence);
    bool ok_nb_c =
        run_ring_variant(name_nb, false, warmup, iters, ring8, cadence, 0u);
    bool ok_bl_c =
        run_ring_variant(name_bl, true, warmup, iters, ring8, cadence, 0u);
    ok_sweep = ok_sweep && ok_nb_c && ok_bl_c;
  }

  printf("\nworkload: ring8 runtime auto-sync sweep (no explicit in-loop sync)\n");
  const size_t auto_sync_vals[] = {0u, 2u, 4u, 8u};
  bool ok_auto = true;
  for (size_t i = 0; i < sizeof(auto_sync_vals) / sizeof(auto_sync_vals[0]);
       ++i) {
    const size_t auto_sync = auto_sync_vals[i];
    char name_nb[128];
    char name_bl[128];
    snprintf(name_nb, sizeof(name_nb), "ring8_auto_non_blocking_%zu", auto_sync);
    snprintf(name_bl, sizeof(name_bl), "ring8_auto_forced_blocking_%zu", auto_sync);
    bool ok_nb_c = run_ring_variant(name_nb, false, warmup, iters, ring8,
                                    0u, auto_sync);
    bool ok_bl_c = run_ring_variant(name_bl, true, warmup, iters, ring8,
                                    0u, auto_sync);
    ok_auto = ok_auto && ok_nb_c && ok_bl_c;
  }

  if (!ok_nb || !ok_bl || !ok_ring_nb || !ok_ring_bl || !ok_sweep ||
      !ok_auto) {
    printf("\nResult: SKIP/INCOMPLETE (see stderr)\n");
    return 0;
  }

  printf("\nResult: compare lines above (lower is better)\n");
  return 0;
}
