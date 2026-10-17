/*
 * Scalar reference versus sv_optim kernels, including the 16,384-element
 * parallelization threshold.
 * Run: bazel run //app/vector:bench_sv_optim
 */

#include "sv_optim.h"

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum BenchOptimizer {
  OPT_SGD,
  OPT_MOMENTUM,
  OPT_RMSPROP,
  OPT_ADAM,
  OPT_CLIP,
} BenchOptimizer;

typedef struct BenchBuffers {
  float *params;
  float *grads;
  float *first;
  float *second;
  float *initial_params;
  float *initial_grads;
  float *initial_first;
  float *initial_second;
  size_t n;
} BenchBuffers;

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
  if (n <= 1000) return 32;
  if (n <= 10000) return 8;
  if (n <= 100000) return 2;
  return 1;
}

static const char *optimizer_name(BenchOptimizer optimizer) {
  switch (optimizer) {
    case OPT_SGD: return "sgd";
    case OPT_MOMENTUM: return "momentum";
    case OPT_RMSPROP: return "rmsprop";
    case OPT_ADAM: return "adam";
    case OPT_CLIP: return "clip_norm";
  }
  return "unknown";
}

static void reset(BenchBuffers *b) {
  const size_t bytes = b->n * sizeof(float);
  memcpy(b->params, b->initial_params, bytes);
  memcpy(b->grads, b->initial_grads, bytes);
  memcpy(b->first, b->initial_first, bytes);
  memcpy(b->second, b->initial_second, bytes);
}

static void scalar_apply(BenchBuffers *b, BenchOptimizer optimizer) {
  const float lr = 0.001f;
  const float grad_scale = 0.75f;
  const float decay = 1.0e-4f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float epsilon = 1.0e-8f;
  const uint64_t step = 5;
  const float inv_b1 = 1.0f / (1.0f - powf(beta1, (float)step));
  const float inv_b2 = 1.0f / (1.0f - powf(beta2, (float)step));

  if (optimizer == OPT_CLIP) {
    double square_sum = 0.0;
    for (size_t i = 0; i < b->n; ++i) {
      const double g = b->grads[i];
      square_sum += g * g;
    }
    const double norm = sqrt(square_sum);
    if (norm > 1.0) {
      const float scale = (float)(1.0 / norm);
      for (size_t i = 0; i < b->n; ++i) b->grads[i] *= scale;
    }
    return;
  }

  for (size_t i = 0; i < b->n; ++i) {
    const float g = b->grads[i] * grad_scale;
    const float old_parameter = b->params[i];
    float parameter = old_parameter - lr * decay * old_parameter;
    switch (optimizer) {
      case OPT_SGD:
        parameter -= lr * g;
        break;
      case OPT_MOMENTUM:
        b->first[i] = 0.9f * b->first[i] + g;
        parameter -= lr * b->first[i];
        break;
      case OPT_RMSPROP:
        b->second[i] = beta2 * b->second[i] + (1.0f - beta2) * g * g;
        parameter -= lr * g / (sqrtf(b->second[i]) + epsilon);
        break;
      case OPT_ADAM:
        b->first[i] = beta1 * b->first[i] + (1.0f - beta1) * g;
        b->second[i] = beta2 * b->second[i] + (1.0f - beta2) * g * g;
        parameter -= lr * b->first[i] * inv_b1 /
                     (sqrtf(b->second[i] * inv_b2) + epsilon);
        break;
      case OPT_CLIP:
        break;
    }
    b->params[i] = parameter;
  }
}

static void forced_parallel_apply(BenchBuffers *b, BenchOptimizer optimizer) {
  const float lr = 0.001f;
  const float grad_scale = 0.75f;
  const float decay = 1.0e-4f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float epsilon = 1.0e-8f;
  const float inv_b1 = 1.0f / (1.0f - powf(beta1, 5.0f));
  const float inv_b2 = 1.0f / (1.0f - powf(beta2, 5.0f));

  if (optimizer == OPT_CLIP) {
    double square_sum = 0.0;
    for (size_t i = 0; i < b->n; ++i) {
      const double g = b->grads[i];
      square_sum += g * g;
    }
    const double norm = sqrt(square_sum);
    if (norm > 1.0) {
      const float scale = (float)(1.0 / norm);
#pragma omp parallel for simd schedule(static)
      for (size_t i = 0; i < b->n; ++i) b->grads[i] *= scale;
    }
    return;
  }

#pragma omp parallel for simd schedule(static)
  for (size_t i = 0; i < b->n; ++i) {
    const float gradient = b->grads[i] * grad_scale;
    const float original = b->params[i];
    float parameter = original - lr * decay * original;
    switch (optimizer) {
      case OPT_SGD:
        parameter -= lr * gradient;
        break;
      case OPT_MOMENTUM:
        b->first[i] = 0.9f * b->first[i] + gradient;
        parameter -= lr * b->first[i];
        break;
      case OPT_RMSPROP:
        b->second[i] = beta2 * b->second[i] +
                       (1.0f - beta2) * gradient * gradient;
        parameter -= lr * gradient / (sqrtf(b->second[i]) + epsilon);
        break;
      case OPT_ADAM:
        b->first[i] = beta1 * b->first[i] + (1.0f - beta1) * gradient;
        b->second[i] = beta2 * b->second[i] +
                       (1.0f - beta2) * gradient * gradient;
        parameter -= lr * b->first[i] * inv_b1 /
                     (sqrtf(b->second[i] * inv_b2) + epsilon);
        break;
      case OPT_CLIP:
        break;
    }
    b->params[i] = parameter;
  }
}

static bool optimized_apply(BenchBuffers *b, BenchOptimizer optimizer) {
  const float lr = 0.001f;
  const float grad_scale = 0.75f;
  const float decay = 1.0e-4f;
  switch (optimizer) {
    case OPT_SGD:
      return sv_optim_sgd(b->params, b->grads, b->n, lr, grad_scale, decay);
    case OPT_MOMENTUM:
      return sv_optim_momentum(b->params, b->grads, b->first, b->n, lr, 0.9f,
                               grad_scale, decay);
    case OPT_RMSPROP:
      return sv_optim_rmsprop(b->params, b->grads, b->second, b->n, lr, 0.999f,
                              1.0e-8f, grad_scale, decay);
    case OPT_ADAM:
      return sv_optim_adam(b->params, b->grads, b->first, b->second, b->n, lr,
                           0.9f, 0.999f, 1.0e-8f, 5, grad_scale, decay);
    case OPT_CLIP:
      return sv_clip_norm(b->grads, b->n, 1.0f) >= 0.0f;
  }
  return false;
}

static double measure(BenchBuffers *buffers, BenchOptimizer optimizer,
                      int mode, size_t reps) {
  double samples[7];
  for (size_t sample = 0; sample < 7; ++sample) {
    reset(buffers);
    const uint64_t start = now_ns();
    for (size_t i = 0; i < reps; ++i) {
      if (mode == 1) {
        if (!optimized_apply(buffers, optimizer)) return -1.0;
      } else if (mode == 2) {
        forced_parallel_apply(buffers, optimizer);
      } else {
        scalar_apply(buffers, optimizer);
      }
    }
    samples[sample] = (double)(now_ns() - start) / 1000.0 / (double)reps;
  }
  return median(samples, 7);
}

static bool allocate_buffers(BenchBuffers *b, size_t n) {
  memset(b, 0, sizeof(*b));
  b->n = n;
  b->params = (float *)malloc(n * sizeof(float));
  b->grads = (float *)malloc(n * sizeof(float));
  b->first = (float *)malloc(n * sizeof(float));
  b->second = (float *)malloc(n * sizeof(float));
  b->initial_params = (float *)malloc(n * sizeof(float));
  b->initial_grads = (float *)malloc(n * sizeof(float));
  b->initial_first = (float *)malloc(n * sizeof(float));
  b->initial_second = (float *)malloc(n * sizeof(float));
  if (!b->params || !b->grads || !b->first || !b->second ||
      !b->initial_params || !b->initial_grads || !b->initial_first ||
      !b->initial_second) {
    return false;
  }
  for (size_t i = 0; i < n; ++i) {
    b->initial_params[i] = 0.1f + (float)(i % 19u) * 0.001f;
    b->initial_grads[i] = (float)((i * 7u) % 37u) * 0.0001f;
    b->initial_first[i] = 0.0f;
    b->initial_second[i] = 0.0f;
  }
  return true;
}

static void destroy_buffers(BenchBuffers *b) {
  free(b->initial_second);
  free(b->initial_first);
  free(b->initial_grads);
  free(b->initial_params);
  free(b->second);
  free(b->first);
  free(b->grads);
  free(b->params);
}

int main(void) {
  const size_t sizes[] = {1000, 10000, 100000, 1000000};
  const BenchOptimizer optimizers[] = {
      OPT_SGD, OPT_MOMENTUM, OPT_RMSPROP, OPT_ADAM, OPT_CLIP};
  const int max_threads = omp_get_max_threads();
  const int thread_counts[] = {1, max_threads};
  omp_set_dynamic(0);

    printf("optimizer,n,threads,scalar_us,sv_optim_us,forced_parallel_us,"
      "api_speedup,parallel_speedup\n");
  for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
    BenchBuffers buffers;
    if (!allocate_buffers(&buffers, sizes[si])) {
      fprintf(stderr, "allocation failed for n=%zu\n", sizes[si]);
      destroy_buffers(&buffers);
      return 1;
    }
    for (size_t ti = 0; ti < 2; ++ti) {
      const int threads = thread_counts[ti];
      if (ti == 1 && threads == thread_counts[0]) continue;
      omp_set_num_threads(threads);
      for (size_t oi = 0; oi < sizeof(optimizers) / sizeof(optimizers[0]);
           ++oi) {
        const size_t reps = repetitions(buffers.n);
         const double before = measure(&buffers, optimizers[oi], 0, reps);
         const double after = measure(&buffers, optimizers[oi], 1, reps);
         const double parallel = measure(&buffers, optimizers[oi], 2, reps);
         printf("%s,%zu,%d,%.3f,%.3f,%.3f,%.2fx,%.2fx\n",
           optimizer_name(optimizers[oi]), buffers.n, threads, before,
           after, parallel, after > 0.0 ? before / after : 0.0,
           parallel > 0.0 ? before / parallel : 0.0);
      }
    }
    destroy_buffers(&buffers);
  }
  omp_set_num_threads(max_threads);
  return 0;
}