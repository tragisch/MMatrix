/*
 * Small dense NN benchmark: GEMM + row bias + activation versus fused API.
 * Run: bazel run //app/matrix:bench_sm_gemm_bias_act
 */

#include "sm.h"

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double median(double *samples, size_t count) {
  for (size_t i = 1; i < count; ++i) {
    const double value = samples[i];
    size_t j = i;
    while (j > 0 && samples[j - 1] > value) {
      samples[j] = samples[j - 1];
      --j;
    }
    samples[j] = value;
  }
  return samples[count / 2];
}

static size_t repetitions(size_t n) {
  if (n <= 32) return 100;
  if (n <= 64) return 24;
  if (n <= 128) return 6;
  return 2;
}

static bool apply_activation(FloatMatrix *out, SmActivation activation) {
  switch (activation) {
    case SM_ACT_NONE: return true;
    case SM_ACT_RELU: return sm_apply_relu(out);
    case SM_ACT_SIGMOID: return sm_apply_sigmoid(out);
    case SM_ACT_TANH: return sm_apply_tanh(out);
  }
  return false;
}

static bool split_path(FloatMatrix *out, const FloatMatrix *a,
                       const FloatMatrix *b, const FloatMatrix *bias,
                       SmActivation activation) {
  if (!sm_gemm(out, 1.0f, a, SM_NO_TRANSPOSE, b, SM_NO_TRANSPOSE, 0.0f)) {
    return false;
  }
  for (size_t row = 0; row < out->rows; ++row) {
    for (size_t col = 0; col < out->cols; ++col) {
      out->values[row * out->cols + col] += bias->values[col];
    }
  }
  return apply_activation(out, activation);
}

static const char *activation_name(SmActivation activation) {
  switch (activation) {
    case SM_ACT_NONE: return "none";
    case SM_ACT_RELU: return "relu";
    case SM_ACT_SIGMOID: return "sigmoid";
    case SM_ACT_TANH: return "tanh";
  }
  return "unknown";
}

static double measure(bool fused, FloatMatrix *out, const FloatMatrix *a,
                      const FloatMatrix *b, const FloatMatrix *bias,
                      SmActivation activation, size_t reps) {
  double samples[9];
  for (size_t sample = 0; sample < 9; ++sample) {
    const uint64_t start = now_ns();
    for (size_t i = 0; i < reps; ++i) {
      const bool ok = fused
          ? sm_gemm_bias_act(out, a, SM_NO_TRANSPOSE, b, SM_NO_TRANSPOSE,
                             bias, activation)
          : split_path(out, a, b, bias, activation);
      if (!ok) return -1.0;
    }
    samples[sample] =
        (double)(now_ns() - start) / 1000.0 / (double)reps;
  }
  return median(samples, 9);
}

int main(void) {
  const size_t sizes[] = {32, 64, 128, 256};
  const SmActivation activations[] = {
      SM_ACT_RELU, SM_ACT_SIGMOID, SM_ACT_TANH};
  const int max_threads = omp_get_max_threads();
  const int thread_counts[] = {1, max_threads};
  omp_set_dynamic(0);

  printf("activation,n,threads,split_us,fused_us,speedup\n");
  for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
    const size_t n = sizes[si];
    FloatMatrix *a = sm_create(n, n);
    FloatMatrix *b = sm_create(n, n);
    FloatMatrix *bias = sm_create(1, n);
    FloatMatrix *out = sm_create(n, n);
    if (!a || !b || !bias || !out) {
      fprintf(stderr, "allocation failed for n=%zu\n", n);
      sm_destroy(a);
      sm_destroy(b);
      sm_destroy(bias);
      sm_destroy(out);
      return 1;
    }
    for (size_t i = 0; i < n * n; ++i) {
      a->values[i] = (float)((i * 17u) % 101u) * 0.01f - 0.5f;
      b->values[i] = (float)((i * 29u) % 97u) * 0.01f - 0.48f;
    }
    for (size_t i = 0; i < n; ++i) {
      bias->values[i] = (float)i * 0.001f - 0.1f;
    }

    for (size_t ti = 0; ti < 2; ++ti) {
      const int threads = thread_counts[ti];
      if (ti == 1 && threads == thread_counts[0]) continue;
      omp_set_num_threads(threads);
      for (size_t ai = 0; ai < sizeof(activations) / sizeof(activations[0]);
           ++ai) {
        const size_t reps = repetitions(n);
        const double split = measure(false, out, a, b, bias, activations[ai],
                                     reps);
        const double fused = measure(true, out, a, b, bias, activations[ai],
                                     reps);
        printf("%s,%zu,%d,%.3f,%.3f,%.2fx\n",
               activation_name(activations[ai]), n, threads, split, fused,
               fused > 0.0 ? split / fused : 0.0);
      }
    }
    sm_destroy(out);
    sm_destroy(bias);
    sm_destroy(b);
    sm_destroy(a);
  }
  omp_set_num_threads(max_threads);
  return 0;
}