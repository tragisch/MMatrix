#include "sv_optim.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpedantic"
#endif
#include <omp.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#define SV_OPTIM_OMP_THRESHOLD 1048576u
#define SV_OPTIM_USE_OMP(N) \
  ((N) > SV_OPTIM_OMP_THRESHOLD && omp_get_max_threads() > 1)

static bool sv_float_is_finite(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7f800000u) != 0x7f800000u;
}

#define SV_OPTIM_LOOP(N, ...)                                                \
  do {                                                                       \
    if (SV_OPTIM_USE_OMP(N)) {                                               \
      _Pragma("omp parallel for simd schedule(static)")                     \
      for (size_t i = 0; i < (N); ++i) {                                     \
        __VA_ARGS__                                                          \
      }                                                                      \
    } else {                                                                 \
      _Pragma("omp simd")                                                   \
      for (size_t i = 0; i < (N); ++i) {                                     \
        __VA_ARGS__                                                          \
      }                                                                      \
    }                                                                        \
  } while (0)

static bool sv_optim_valid_base(const float *params, const float *grads,
                                size_t n, float learning_rate,
                                float grad_scale, float weight_decay) {
  return params != NULL && grads != NULL && n > 0 &&
      sv_float_is_finite(learning_rate) && learning_rate > 0.0f &&
      sv_float_is_finite(weight_decay) && weight_decay >= 0.0f &&
      sv_float_is_finite(grad_scale);
}

bool sv_optim_sgd(float *params, const float *grads, size_t n,
                  float learning_rate, float grad_scale, float weight_decay) {
  if (!sv_optim_valid_base(params, grads, n, learning_rate, grad_scale,
                           weight_decay)) {
    return false;
  }

  SV_OPTIM_LOOP(n,
    float parameter = params[i] - learning_rate * weight_decay * params[i];
    parameter -= learning_rate * (grads[i] * grad_scale);
    params[i] = parameter;
  );
  return true;
}

bool sv_optim_momentum(float *params, const float *grads,
                       float *momentum_buffer, size_t n, float learning_rate,
                       float momentum, float grad_scale, float weight_decay) {
  if (!sv_optim_valid_base(params, grads, n, learning_rate, grad_scale,
                           weight_decay) ||
      momentum_buffer == NULL || !sv_float_is_finite(momentum) ||
      momentum < 0.0f || momentum >= 1.0f) {
    return false;
  }

  SV_OPTIM_LOOP(n,
    const float gradient = grads[i] * grad_scale;
    const float velocity = momentum * momentum_buffer[i] + gradient;
    momentum_buffer[i] = velocity;
    float parameter = params[i] - learning_rate * weight_decay * params[i];
    parameter -= learning_rate * velocity;
    params[i] = parameter;
  );
  return true;
}

bool sv_optim_rmsprop(float *params, const float *grads,
                      float *squared_grad_buffer, size_t n,
                      float learning_rate, float beta2, float epsilon,
                      float grad_scale, float weight_decay) {
  if (!sv_optim_valid_base(params, grads, n, learning_rate, grad_scale,
                           weight_decay) ||
      squared_grad_buffer == NULL || !sv_float_is_finite(beta2) ||
      !sv_float_is_finite(epsilon) || beta2 <= 0.0f || beta2 >= 1.0f ||
      epsilon <= 0.0f) {
    return false;
  }

  SV_OPTIM_LOOP(n,
    const float gradient = grads[i] * grad_scale;
    const float square = beta2 * squared_grad_buffer[i] +
                         (1.0f - beta2) * gradient * gradient;
    squared_grad_buffer[i] = square;
    float parameter = params[i] - learning_rate * weight_decay * params[i];
    parameter -= learning_rate * gradient / (sqrtf(square) + epsilon);
    params[i] = parameter;
  );
  return true;
}

bool sv_optim_adam(float *params, const float *grads, float *first_moment,
                   float *second_moment, size_t n, float learning_rate,
                   float beta1, float beta2, float epsilon, uint64_t step,
                   float grad_scale, float weight_decay) {
  if (!sv_optim_valid_base(params, grads, n, learning_rate, grad_scale,
                           weight_decay) ||
      first_moment == NULL || second_moment == NULL ||
      !sv_float_is_finite(beta1) || !sv_float_is_finite(beta2) ||
      !sv_float_is_finite(epsilon) || beta1 <= 0.0f || beta1 >= 1.0f ||
      beta2 <= 0.0f || beta2 >= 1.0f || epsilon <= 0.0f || step == 0) {
    return false;
  }

  const float inverse_bias1 = 1.0f / (1.0f - powf(beta1, (float)step));
  const float inverse_bias2 = 1.0f / (1.0f - powf(beta2, (float)step));
  SV_OPTIM_LOOP(n,
    const float gradient = grads[i] * grad_scale;
    const float first = beta1 * first_moment[i] + (1.0f - beta1) * gradient;
    const float second = beta2 * second_moment[i] +
                         (1.0f - beta2) * gradient * gradient;
    first_moment[i] = first;
    second_moment[i] = second;
    const float corrected_first = first * inverse_bias1;
    const float corrected_second = second * inverse_bias2;
    float parameter = params[i] - learning_rate * weight_decay * params[i];
    parameter -= learning_rate * corrected_first /
                 (sqrtf(corrected_second) + epsilon);
    params[i] = parameter;
  );
  return true;
}

float sv_clip_norm(float *grads, size_t n, float max_norm) {
  if (!sv_float_is_finite(max_norm) || max_norm < 0.0f ||
      (n > 0 && grads == NULL)) {
    return -1.0f;
  }
  if (n == 0) {
    return 0.0f;
  }

  double square_sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double value = grads[i];
    square_sum += value * value;
  }
  const double norm = sqrt(square_sum);
  if (norm > (double)max_norm && norm > 0.0) {
    const float scale = (float)((double)max_norm / norm);
    SV_OPTIM_LOOP(n,
      grads[i] *= scale;
    );
  }
  return (float)norm;
}