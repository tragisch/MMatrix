/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * bench_st_layout — measure NCHW<->NHWC materialization cost versus Conv2D.
 *
 * Purpose:
 *   Provide a concrete baseline for layout work (Point 5):
 *   - cost of NCHW -> NHWC materialization (permute view + clone)
 *   - cost of NHWC -> NCHW materialization
 *   - roundtrip cost relative to Conv2D AUTO runtime
 */

#include "st.h"
#include "st_conv.h"
#include "log.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double ns_to_ms(uint64_t ns, size_t iters) {
  return (double)ns / 1000000.0 / (double)iters;
}

static FloatTensor *make4d(size_t n, size_t c, size_t h, size_t w) {
  const size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

static void fill_rand(FloatTensor *t, uint32_t seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1664525u + 1013904223u;
    t->values[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

typedef struct LayoutCase {
  const char *name;
  size_t n, c_in, c_out, h, w, k;
  size_t warmup, iters;
} LayoutCase;

static bool run_case(const LayoutCase *cfg) {
  if (!cfg) return false;

  FloatTensor *input = make4d(cfg->n, cfg->c_in, cfg->h, cfg->w);
  FloatTensor *weight = make4d(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
  if (!input || !weight) {
    st_destroy(weight);
    st_destroy(input);
    return false;
  }
  fill_rand(input, 11u);
  fill_rand(weight, 22u);

  StConv2dParams p = st_conv2d_default_params();
  p.backend = ST_CONV_BACKEND_AUTO;
  p.pad_h = p.pad_w = (cfg->k == 3) ? 1u : 0u;

  size_t out_h = 0, out_w = 0;
  if (!st_conv2d_output_hw(cfg->h, cfg->w, cfg->k, cfg->k, &p, &out_h, &out_w)) {
    st_destroy(weight);
    st_destroy(input);
    return false;
  }

  FloatTensor *out = make4d(cfg->n, cfg->c_out, out_h, out_w);
  if (!out) {
    st_destroy(weight);
    st_destroy(input);
    return false;
  }

  const size_t perm_nhwc[4] = {0, 2, 3, 1};
  const size_t perm_nchw[4] = {0, 3, 1, 2};

  /* Warmup conv + layout transforms */
  for (size_t i = 0; i < cfg->warmup; ++i) {
    if (!st_conv2d_nchw(input, weight, NULL, &p, out)) {
      st_destroy(out);
      st_destroy(weight);
      st_destroy(input);
      return false;
    }
    st_tensor_sync(out);

    FloatTensor *v1 = st_permute_view(input, perm_nhwc);
    FloatTensor *nhwc = v1 ? st_clone(v1) : NULL;
    FloatTensor *v2 = nhwc ? st_permute_view(nhwc, perm_nchw) : NULL;
    FloatTensor *nchw = v2 ? st_clone(v2) : NULL;
    st_destroy(nchw);
    st_destroy(v2);
    st_destroy(nhwc);
    st_destroy(v1);
  }

  uint64_t conv_ns = 0;
  {
    const uint64_t t0 = now_ns();
    for (size_t i = 0; i < cfg->iters; ++i) {
      if (!st_conv2d_nchw(input, weight, NULL, &p, out)) {
        st_destroy(out);
        st_destroy(weight);
        st_destroy(input);
        return false;
      }
      st_tensor_sync(out);
    }
    const uint64_t t1 = now_ns();
    conv_ns = t1 - t0;
  }

  uint64_t to_nhwc_ns = 0;
  {
    const uint64_t t0 = now_ns();
    for (size_t i = 0; i < cfg->iters; ++i) {
      FloatTensor *v1 = st_permute_view(input, perm_nhwc);
      FloatTensor *nhwc = v1 ? st_clone(v1) : NULL;
      st_destroy(nhwc);
      st_destroy(v1);
      if (!nhwc) {
        st_destroy(out);
        st_destroy(weight);
        st_destroy(input);
        return false;
      }
    }
    const uint64_t t1 = now_ns();
    to_nhwc_ns = t1 - t0;
  }

  FloatTensor *nhwc_src = NULL;
  {
    FloatTensor *v1 = st_permute_view(input, perm_nhwc);
    nhwc_src = v1 ? st_clone(v1) : NULL;
    st_destroy(v1);
    if (!nhwc_src) {
      st_destroy(out);
      st_destroy(weight);
      st_destroy(input);
      return false;
    }
  }

  uint64_t to_nchw_ns = 0;
  {
    const uint64_t t0 = now_ns();
    for (size_t i = 0; i < cfg->iters; ++i) {
      FloatTensor *v2 = st_permute_view(nhwc_src, perm_nchw);
      FloatTensor *nchw = v2 ? st_clone(v2) : NULL;
      st_destroy(nchw);
      st_destroy(v2);
      if (!nchw) {
        st_destroy(nhwc_src);
        st_destroy(out);
        st_destroy(weight);
        st_destroy(input);
        return false;
      }
    }
    const uint64_t t1 = now_ns();
    to_nchw_ns = t1 - t0;
  }

  const double conv_ms = ns_to_ms(conv_ns, cfg->iters);
  const double to_nhwc_ms = ns_to_ms(to_nhwc_ns, cfg->iters);
  const double to_nchw_ms = ns_to_ms(to_nchw_ns, cfg->iters);
  const double roundtrip_ms = to_nhwc_ms + to_nchw_ms;
  const double overhead_pct = (conv_ms > 0.0) ? (roundtrip_ms / conv_ms) * 100.0 : 0.0;

  printf("layout,%s,%zu,%zu,%zu,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.2f\n",
         cfg->name, cfg->n, cfg->c_in, cfg->c_out, cfg->h, cfg->w,
         conv_ms, to_nhwc_ms, to_nchw_ms, roundtrip_ms, overhead_pct);

  st_destroy(nhwc_src);
  st_destroy(out);
  st_destroy(weight);
  st_destroy(input);
  return true;
}

int main(void) {
  log_set_level(LOG_WARN);
  printf("suite,case_name,n,c_in,c_out,h,w,conv_auto_ms,layout_to_nhwc_ms,layout_to_nchw_ms,layout_roundtrip_ms,layout_overhead_pct_of_conv\n");

  static const LayoutCase cases[] = {
      {"conv_small", 1, 8, 8, 16, 16, 3, 2, 30},
      {"conv_medium", 4, 32, 64, 56, 56, 3, 2, 10},
      {"resnet_s1", 1, 64, 64, 56, 56, 3, 2, 10},
      {"pw_medium", 4, 64, 128, 56, 56, 1, 2, 10},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    if (!run_case(&cases[i])) {
      fprintf(stderr, "layout benchmark failed for case: %s\n", cases[i].name);
      return 1;
    }
  }

  return 0;
}
