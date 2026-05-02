#include "st_conv.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct BenchCase {
  const char *name;
  size_t n;
  size_t c_in;
  size_t c_out;
  size_t h;
  size_t w;
  size_t k;
  size_t stride;
  size_t pad;
  size_t iters;
} BenchCase;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static FloatTensor *create_4d(size_t n, size_t c, size_t h, size_t w) {
  size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

static void fill_rand(FloatTensor *t, uint32_t seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1664525u + 1013904223u;
    t->values[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

static double elapsed_ms(uint64_t start_ns, uint64_t end_ns, size_t iters) {
  return (double)(end_ns - start_ns) / 1000000.0 / (double)iters;
}

static bool bench_case(const BenchCase *cfg) {
  size_t out_h = 0;
  size_t out_w = 0;
  StConv2dParams p = st_conv2d_default_params();
  p.stride_h = cfg->stride;
  p.stride_w = cfg->stride;
  p.pad_h = cfg->pad;
  p.pad_w = cfg->pad;
  p.backend = ST_CONV_BACKEND_GEMM;

  if (!st_conv2d_output_hw(cfg->h, cfg->w, cfg->k, cfg->k, &p, &out_h,
                           &out_w)) {
    return false;
  }

  FloatTensor *input = create_4d(cfg->n, cfg->c_in, cfg->h, cfg->w);
  FloatTensor *weight = create_4d(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
  FloatTensor *grad_output = create_4d(cfg->n, cfg->c_out, out_h, out_w);
  FloatTensor *grad_input = create_4d(cfg->n, cfg->c_in, cfg->h, cfg->w);
  FloatTensor *grad_weight = create_4d(cfg->c_out, cfg->c_in, cfg->k, cfg->k);
  if (!input || !weight || !grad_output || !grad_input || !grad_weight) {
    st_destroy(grad_weight);
    st_destroy(grad_input);
    st_destroy(grad_output);
    st_destroy(weight);
    st_destroy(input);
    return false;
  }

  fill_rand(input, 11u);
  fill_rand(weight, 22u);
  fill_rand(grad_output, 33u);

  const size_t out_spatial = out_h * out_w;
  const size_t patch_size = cfg->c_in * cfg->k * cfg->k;
  const double col_mib =
      (double)(out_spatial * patch_size * sizeof(float)) / (1024.0 * 1024.0);

  for (size_t i = 0; i < 3; ++i) {
    if (!st_conv2d_backward_data_nchw(grad_output, weight, &p, grad_input) ||
        !st_conv2d_backward_weight_nchw(input, grad_output, &p, grad_weight)) {
      st_destroy(grad_weight);
      st_destroy(grad_input);
      st_destroy(grad_output);
      st_destroy(weight);
      st_destroy(input);
      return false;
    }
  }

  uint64_t start = now_ns();
  for (size_t i = 0; i < cfg->iters; ++i) {
    if (!st_conv2d_backward_data_nchw(grad_output, weight, &p, grad_input)) {
      st_destroy(grad_weight);
      st_destroy(grad_input);
      st_destroy(grad_output);
      st_destroy(weight);
      st_destroy(input);
      return false;
    }
  }
  uint64_t data_end = now_ns();

  for (size_t i = 0; i < cfg->iters; ++i) {
    if (!st_conv2d_backward_weight_nchw(input, grad_output, &p, grad_weight)) {
      st_destroy(grad_weight);
      st_destroy(grad_input);
      st_destroy(grad_output);
      st_destroy(weight);
      st_destroy(input);
      return false;
    }
  }
  uint64_t weight_end = now_ns();

  printf("%s\n", cfg->name);
  printf("  shape: N=%zu Cin=%zu Cout=%zu H=%zu W=%zu K=%zu stride=%zu pad=%zu\n",
         cfg->n, cfg->c_in, cfg->c_out, cfg->h, cfg->w, cfg->k, cfg->stride,
         cfg->pad);
  printf("  out: %zux%zu  col_buf: %.2f MiB\n", out_h, out_w, col_mib);
  printf("  backward_data_gemm:   %.3f ms/iter\n",
         elapsed_ms(start, data_end, cfg->iters));
  printf("  backward_weight_gemm: %.3f ms/iter\n",
         elapsed_ms(data_end, weight_end, cfg->iters));

  st_destroy(grad_weight);
  st_destroy(grad_input);
  st_destroy(grad_output);
  st_destroy(weight);
  st_destroy(input);
  return true;
}

static bool bench_reference_small(void) {
  const BenchCase cfg = {
      .name = "reference-small",
      .n = 2,
      .c_in = 8,
      .c_out = 8,
      .h = 16,
      .w = 16,
      .k = 3,
      .stride = 1,
      .pad = 1,
      .iters = 5,
  };

  size_t out_h = 0;
  size_t out_w = 0;
  StConv2dParams p = st_conv2d_default_params();
  p.stride_h = cfg.stride;
  p.stride_w = cfg.stride;
  p.pad_h = cfg.pad;
  p.pad_w = cfg.pad;
  p.backend = ST_CONV_BACKEND_REFERENCE;

  if (!st_conv2d_output_hw(cfg.h, cfg.w, cfg.k, cfg.k, &p, &out_h, &out_w)) {
    return false;
  }

  FloatTensor *input = create_4d(cfg.n, cfg.c_in, cfg.h, cfg.w);
  FloatTensor *weight = create_4d(cfg.c_out, cfg.c_in, cfg.k, cfg.k);
  FloatTensor *grad_output = create_4d(cfg.n, cfg.c_out, out_h, out_w);
  FloatTensor *grad_input = create_4d(cfg.n, cfg.c_in, cfg.h, cfg.w);
  FloatTensor *grad_weight = create_4d(cfg.c_out, cfg.c_in, cfg.k, cfg.k);
  if (!input || !weight || !grad_output || !grad_input || !grad_weight) {
    st_destroy(grad_weight);
    st_destroy(grad_input);
    st_destroy(grad_output);
    st_destroy(weight);
    st_destroy(input);
    return false;
  }

  fill_rand(input, 101u);
  fill_rand(weight, 202u);
  fill_rand(grad_output, 303u);

  uint64_t start = now_ns();
  for (size_t i = 0; i < cfg.iters; ++i) {
    if (!st_conv2d_backward_data_nchw(grad_output, weight, &p, grad_input)) {
      return false;
    }
  }
  uint64_t data_end = now_ns();

  for (size_t i = 0; i < cfg.iters; ++i) {
    if (!st_conv2d_backward_weight_nchw(input, grad_output, &p, grad_weight)) {
      return false;
    }
  }
  uint64_t weight_end = now_ns();

  printf("%s\n", cfg.name);
  printf("  backward_data_ref:    %.3f ms/iter\n",
         elapsed_ms(start, data_end, cfg.iters));
  printf("  backward_weight_ref:  %.3f ms/iter\n",
         elapsed_ms(data_end, weight_end, cfg.iters));

  st_destroy(grad_weight);
  st_destroy(grad_input);
  st_destroy(grad_output);
  st_destroy(weight);
  st_destroy(input);
  return true;
}

int main(void) {
  const BenchCase cases[] = {
      {
          .name = "gemm-k3-resnetish",
          .n = 8,
          .c_in = 32,
          .c_out = 64,
          .h = 56,
          .w = 56,
          .k = 3,
          .stride = 1,
          .pad = 1,
          .iters = 20,
      },
      {
          .name = "gemm-k1-bottleneck",
          .n = 8,
          .c_in = 64,
          .c_out = 64,
          .h = 56,
          .w = 56,
          .k = 1,
          .stride = 1,
          .pad = 0,
          .iters = 20,
      },
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    if (!bench_case(&cases[i])) {
      fprintf(stderr, "benchmark failed: %s\n", cases[i].name);
      return 1;
    }
  }

  if (!bench_reference_small()) {
    fprintf(stderr, "benchmark failed: reference-small\n");
    return 1;
  }

  return 0;
}
