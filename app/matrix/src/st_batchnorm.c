/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_batchnorm.h"

#include <log.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- Validation helpers ---- */

static bool st_bn_is_valid_4d(const FloatTensor *t) {
  return t != NULL && t->values != NULL && t->ndim == 4 &&
         st_is_contiguous(t);
}

static bool st_bn_is_valid_1d(const FloatTensor *t, size_t expected_len) {
  return t != NULL && t->values != NULL && t->ndim == 1 &&
         t->shape[0] == expected_len && st_is_contiguous(t);
}

/* ---- Batch Normalization 2D Forward ---- */

bool st_batchnorm2d_forward(const FloatTensor *input,
                            const FloatTensor *gamma,
                            const FloatTensor *beta, float epsilon,
                            FloatTensor *output, FloatTensor *mean,
                            FloatTensor *var) {
  if (!st_bn_is_valid_4d(input) || !st_bn_is_valid_4d(output)) {
    log_error(
        "Error: st_batchnorm2d_forward expects valid contiguous 4D tensors.");
    return false;
  }

  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];
  const size_t spatial = h * w;
  const size_t m = n * spatial; /* number of samples per channel */

  if (output->shape[0] != n || output->shape[1] != c ||
      output->shape[2] != h || output->shape[3] != w) {
    log_error("Error: st_batchnorm2d_forward output shape mismatch.");
    return false;
  }

  if (!st_bn_is_valid_1d(mean, c) || !st_bn_is_valid_1d(var, c)) {
    log_error(
        "Error: st_batchnorm2d_forward mean/var must be contiguous [C].");
    return false;
  }

  if (gamma && !st_bn_is_valid_1d(gamma, c)) {
    log_error("Error: st_batchnorm2d_forward gamma must be contiguous [C].");
    return false;
  }
  if (beta && !st_bn_is_valid_1d(beta, c)) {
    log_error("Error: st_batchnorm2d_forward beta must be contiguous [C].");
    return false;
  }

  /* Step 1: Compute per-channel mean. */
  memset(mean->values, 0, c * sizeof(float));
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *plane = input->values + (ni * c + ci) * spatial;
      float sum = 0.0f;
      for (size_t i = 0; i < spatial; ++i) {
        sum += plane[i];
      }
      mean->values[ci] += sum;
    }
  }
  for (size_t ci = 0; ci < c; ++ci) {
    mean->values[ci] /= (float)m;
  }

  /* Step 2: Compute per-channel variance. */
  memset(var->values, 0, c * sizeof(float));
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *plane = input->values + (ni * c + ci) * spatial;
      const float mu = mean->values[ci];
      float sum_sq = 0.0f;
      for (size_t i = 0; i < spatial; ++i) {
        const float diff = plane[i] - mu;
        sum_sq += diff * diff;
      }
      var->values[ci] += sum_sq;
    }
  }
  for (size_t ci = 0; ci < c; ++ci) {
    var->values[ci] /= (float)m;
  }

  /* Step 3: Normalize, scale, shift. */
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      float *out_plane = output->values + (ni * c + ci) * spatial;
      const float mu = mean->values[ci];
      const float inv_std = 1.0f / sqrtf(var->values[ci] + epsilon);
      const float g = gamma ? gamma->values[ci] : 1.0f;
      const float b = beta ? beta->values[ci] : 0.0f;

      for (size_t i = 0; i < spatial; ++i) {
        out_plane[i] = (in_plane[i] - mu) * inv_std * g + b;
      }
    }
  }

  return true;
}

/* ---- Batch Normalization 2D Backward ---- */

bool st_batchnorm2d_backward(const FloatTensor *grad_output,
                             const FloatTensor *input,
                             const FloatTensor *mean, const FloatTensor *var,
                             const FloatTensor *gamma, float epsilon,
                             FloatTensor *grad_input, FloatTensor *grad_gamma,
                             FloatTensor *grad_beta) {
  if (!st_bn_is_valid_4d(grad_output) || !st_bn_is_valid_4d(input) ||
      !st_bn_is_valid_4d(grad_input)) {
    log_error(
        "Error: st_batchnorm2d_backward expects valid contiguous 4D tensors.");
    return false;
  }

  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];
  const size_t spatial = h * w;
  const size_t m = n * spatial;

  if (!st_bn_is_valid_1d(mean, c) || !st_bn_is_valid_1d(var, c)) {
    log_error(
        "Error: st_batchnorm2d_backward mean/var must be contiguous [C].");
    return false;
  }

  if (gamma && !st_bn_is_valid_1d(gamma, c)) {
    log_error("Error: st_batchnorm2d_backward gamma must be contiguous [C].");
    return false;
  }

  if (grad_gamma && !st_bn_is_valid_1d(grad_gamma, c)) {
    log_error(
        "Error: st_batchnorm2d_backward grad_gamma must be contiguous [C].");
    return false;
  }
  if (grad_beta && !st_bn_is_valid_1d(grad_beta, c)) {
    log_error(
        "Error: st_batchnorm2d_backward grad_beta must be contiguous [C].");
    return false;
  }

  /* Allocate temp buffers for per-channel intermediate sums. */
  float *sum_dy = (float *)calloc(c, sizeof(float));
  float *sum_dy_xhat = (float *)calloc(c, sizeof(float));
  if (!sum_dy || !sum_dy_xhat) {
    free(sum_dy);
    free(sum_dy_xhat);
    log_error("Error: st_batchnorm2d_backward allocation failed.");
    return false;
  }

  /* Pass 1: accumulate sum(dy) and sum(dy * x_hat) per channel,
   * and gradients w.r.t. gamma and beta. */
  if (grad_gamma) {
    memset(grad_gamma->values, 0, c * sizeof(float));
  }
  if (grad_beta) {
    memset(grad_beta->values, 0, c * sizeof(float));
  }

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *go_plane = grad_output->values + (ni * c + ci) * spatial;
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      const float mu = mean->values[ci];
      const float inv_std = 1.0f / sqrtf(var->values[ci] + epsilon);
      const float g = gamma ? gamma->values[ci] : 1.0f;

      for (size_t i = 0; i < spatial; ++i) {
        const float dy = go_plane[i];
        const float x_hat = (in_plane[i] - mu) * inv_std;

        sum_dy[ci] += dy * g;
        sum_dy_xhat[ci] += dy * g * x_hat;

        if (grad_gamma) {
          grad_gamma->values[ci] += dy * x_hat;
        }
        if (grad_beta) {
          grad_beta->values[ci] += dy;
        }
      }
    }
  }

  /* Pass 2: compute grad_input.
   * dx_i = inv_std / m * (m * dy_i*gamma - sum_dy - x_hat_i * sum_dy_xhat) */
  const float inv_m = 1.0f / (float)m;
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *go_plane = grad_output->values + (ni * c + ci) * spatial;
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      float *gi_plane = grad_input->values + (ni * c + ci) * spatial;
      const float mu = mean->values[ci];
      const float inv_std = 1.0f / sqrtf(var->values[ci] + epsilon);
      const float g = gamma ? gamma->values[ci] : 1.0f;

      for (size_t i = 0; i < spatial; ++i) {
        const float x_hat = (in_plane[i] - mu) * inv_std;
        gi_plane[i] = inv_std * inv_m *
                      ((float)m * go_plane[i] * g - sum_dy[ci] -
                       x_hat * sum_dy_xhat[ci]);
      }
    }
  }

  free(sum_dy);
  free(sum_dy_xhat);
  return true;
}
