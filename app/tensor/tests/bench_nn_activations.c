/*
 * Median benchmark for scalar NN activations versus mMatrix sm/st kernels.
 * Run: bazel run //app/tensor:bench_nn_activations
 */

#include "sm.h"
#include "st.h"

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum BenchActivation {
  BENCH_SIGMOID,
  BENCH_TANH,
  BENCH_GELU_TANH,
  BENCH_GELU_ERF,
  BENCH_SOFTMAX,
} BenchActivation;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static float scalar_gelu_tanh(float x) {
  return 0.5f * x *
         (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static float scalar_apply(BenchActivation activation, float x) {
  switch (activation) {
    case BENCH_SIGMOID:
      return 1.0f / (1.0f + expf(-x));
    case BENCH_TANH:
      return tanhf(x);
    case BENCH_GELU_TANH:
      return scalar_gelu_tanh(x);
    case BENCH_GELU_ERF:
      return 0.5f * x * (1.0f + erff(x * 0.7071067812f));
    case BENCH_SOFTMAX:
      break;
  }
  return x;
}

static const char *activation_name(BenchActivation activation) {
  switch (activation) {
    case BENCH_SIGMOID: return "sigmoid";
    case BENCH_TANH: return "tanh";
    case BENCH_GELU_TANH: return "gelu_tanh";
    case BENCH_GELU_ERF: return "gelu_erf";
    case BENCH_SOFTMAX: return "softmax";
  }
  return "unknown";
}

static void scalar_softmax(float *values, size_t rows, size_t cols) {
  for (size_t row = 0; row < rows; ++row) {
    float *x = values + row * cols;
    float maximum = x[0];
    for (size_t col = 1; col < cols; ++col) {
      if (x[col] > maximum) maximum = x[col];
    }
    float sum = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
      x[col] = expf(x[col] - maximum);
      sum += x[col];
    }
    for (size_t col = 0; col < cols; ++col) x[col] /= sum;
  }
}

static bool apply_sm(FloatMatrix *mat, BenchActivation activation) {
  switch (activation) {
    case BENCH_SIGMOID: return sm_apply_sigmoid(mat);
    case BENCH_TANH: return sm_apply_tanh(mat);
    case BENCH_GELU_TANH: return sm_apply_gelu_tanh(mat);
    case BENCH_GELU_ERF: return sm_apply_gelu_erf(mat);
    case BENCH_SOFTMAX: return sm_apply_softmax(mat);
  }
  return false;
}

static bool apply_st(FloatTensor *tensor, BenchActivation activation) {
  switch (activation) {
    case BENCH_SIGMOID: return st_apply_sigmoid(tensor);
    case BENCH_TANH: return st_apply_tanh(tensor);
    case BENCH_GELU_TANH: return st_apply_gelu_tanh(tensor);
    case BENCH_GELU_ERF: return st_apply_gelu_erf(tensor);
    case BENCH_SOFTMAX: return st_apply_softmax(tensor);
  }
  return false;
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

static size_t repetitions_for(size_t n) {
  if (n <= 256) return 4096;
  if (n <= 1024) return 2048;
  if (n <= 10000) return 256;
  if (n <= 100000) return 32;
  return 4;
}

static double measure_scalar(const float *input, float *work, size_t rows,
                             size_t cols, BenchActivation activation,
                             size_t repetitions) {
  enum { SAMPLES = 9 };
  double samples[SAMPLES];
  const size_t n = rows * cols;
  for (size_t s = 0; s < SAMPLES; ++s) {
    memcpy(work, input, n * sizeof(float));
    const uint64_t start = now_ns();
    for (size_t r = 0; r < repetitions; ++r) {
      if (activation == BENCH_SOFTMAX) {
        scalar_softmax(work, rows, cols);
      } else {
        for (size_t i = 0; i < n; ++i) {
          work[i] = scalar_apply(activation, work[i]);
        }
      }
    }
    samples[s] = (double)(now_ns() - start) / 1000.0 / (double)repetitions;
  }
  return median(samples, SAMPLES);
}

static double measure_sm(const float *input, FloatMatrix *work,
                         BenchActivation activation, size_t repetitions) {
  enum { SAMPLES = 9 };
  double samples[SAMPLES];
  const size_t n = work->rows * work->cols;
  for (size_t s = 0; s < SAMPLES; ++s) {
    memcpy(work->values, input, n * sizeof(float));
    const uint64_t start = now_ns();
    for (size_t r = 0; r < repetitions; ++r) {
      if (!apply_sm(work, activation)) return -1.0;
    }
    samples[s] = (double)(now_ns() - start) / 1000.0 / (double)repetitions;
  }
  return median(samples, SAMPLES);
}

static double measure_st(const float *input, FloatTensor *work,
                         BenchActivation activation, size_t repetitions) {
  enum { SAMPLES = 9 };
  double samples[SAMPLES];
  const size_t n = work->numel;
  for (size_t s = 0; s < SAMPLES; ++s) {
    memcpy(work->values, input, n * sizeof(float));
    const uint64_t start = now_ns();
    for (size_t r = 0; r < repetitions; ++r) {
      if (!apply_st(work, activation)) return -1.0;
    }
    samples[s] = (double)(now_ns() - start) / 1000.0 / (double)repetitions;
  }
  return median(samples, SAMPLES);
}

int main(void) {
  const size_t sizes[] = {256, 1024, 10000, 100000, 1000000};
  const BenchActivation activations[] = {
      BENCH_SIGMOID, BENCH_TANH, BENCH_GELU_TANH, BENCH_GELU_ERF,
      BENCH_SOFTMAX};
  const int default_threads = omp_get_max_threads();
  omp_set_dynamic(0);

  printf("activation,n,threads,scalar_us,sm_us,st_us,sm_speedup,st_speedup\n");
  for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
    const size_t n = sizes[si];
    const size_t cols = n / 4;
    const size_t rows = n / cols;
    float *input = (float *)malloc(n * sizeof(float));
    float *scalar_work = (float *)malloc(n * sizeof(float));
    FloatMatrix *sm = sm_create(rows, cols);
    const size_t shape[2] = {rows, cols};
    FloatTensor *st = st_create(2, shape);
    if (!input || !scalar_work || !sm || !st) {
      fprintf(stderr, "allocation failed for n=%zu\n", n);
      free(input);
      free(scalar_work);
      sm_destroy(sm);
      st_destroy(st);
      return 1;
    }
    for (size_t i = 0; i < n; ++i) {
      input[i] = sinf((float)i * 0.013f) * 4.0f;
    }

    const int thread_counts[2] = {1, default_threads};
    for (size_t ti = 0; ti < 2; ++ti) {
      const int threads = thread_counts[ti];
      if (ti == 1 && threads == thread_counts[0]) continue;
      omp_set_num_threads(threads);
      for (size_t ai = 0; ai < sizeof(activations) / sizeof(activations[0]);
           ++ai) {
        const BenchActivation activation = activations[ai];
        const size_t repetitions = repetitions_for(n);
        const double before = measure_scalar(input, scalar_work, rows, cols,
                      activation, repetitions);
        const double sm_us = measure_sm(input, sm, activation, repetitions);
        const double st_us = measure_st(input, st, activation, repetitions);
        printf("%s,%zu,%d,%.3f,%.3f,%.3f,%.2fx,%.2fx\n",
               activation_name(activation), n, threads, before, sm_us, st_us,
               sm_us > 0.0 ? before / sm_us : 0.0,
               st_us > 0.0 ? before / st_us : 0.0);
      }
    }
    sm_destroy(sm);
    st_destroy(st);
    free(input);
    free(scalar_work);
  }
  omp_set_num_threads(default_threads);
  return 0;
}