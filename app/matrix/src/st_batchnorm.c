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

#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
#include <Accelerate/Accelerate.h>
#endif

#if defined(USE_ACCELERATE_MPS) && defined(__APPLE__)
#include "st_mps.h"
#define ST_BN_MPS_THRESHOLD 4096 /* numel threshold for MPS dispatch */
#endif

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

#if defined(USE_ACCELERATE_MPS) && defined(__APPLE__)
  /* MPS dispatch for large tensors. */
  if (input->numel >= ST_BN_MPS_THRESHOLD) {
    bool ok = st_batchnorm2d_forward_mps(
        input->values, n, c, h, w,
        gamma ? gamma->values : NULL,
        beta ? beta->values : NULL,
        epsilon, output->values, mean->values, var->values);
    if (ok) return true;
    /* Fallback to CPU on MPS failure. */
  }
#endif

  const float inv_m = 1.0f / (float)m;

  /* Step 1: Compute per-channel mean. */
  memset(mean->values, 0, c * sizeof(float));
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *plane = input->values + (ni * c + ci) * spatial;
#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
      float plane_sum = 0.0f;
      vDSP_sve(plane, 1, &plane_sum, (vDSP_Length)spatial);
      mean->values[ci] += plane_sum;
#else
      float sum = 0.0f;
      for (size_t i = 0; i < spatial; ++i) {
        sum += plane[i];
      }
      mean->values[ci] += sum;
#endif
    }
  }
  for (size_t ci = 0; ci < c; ++ci) {
    mean->values[ci] *= inv_m;
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
    var->values[ci] *= inv_m;
  }

  /* Pre-compute inv_std per channel (avoid repeated sqrtf). */
  float *inv_std = (float *)malloc(c * sizeof(float));
  if (!inv_std) {
    log_error("Error: st_batchnorm2d_forward allocation failed.");
    return false;
  }
  for (size_t ci = 0; ci < c; ++ci) {
    inv_std[ci] = 1.0f / sqrtf(var->values[ci] + epsilon);
  }

  /* Step 3: Normalize, scale, shift — parallelized over N*C planes. */
  const size_t nc = n * c;

#pragma omp parallel for schedule(static) if (nc > 4)
  for (size_t nci = 0; nci < nc; ++nci) {
    const size_t ni = nci / c;
    const size_t ci = nci % c;
    const float *in_plane = input->values + (ni * c + ci) * spatial;
    float *out_plane = output->values + (ni * c + ci) * spatial;
    const float mu = mean->values[ci];
    const float is = inv_std[ci];
    const float g = gamma ? gamma->values[ci] : 1.0f;
    const float b = beta ? beta->values[ci] : 0.0f;
    const float g_is = g * is; /* fuse multiply */

    for (size_t i = 0; i < spatial; ++i) {
      out_plane[i] = (in_plane[i] - mu) * g_is + b;
    }
  }

  free(inv_std);
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

  /* Pre-compute inv_std per channel (avoid repeated sqrtf). */
  float *inv_std_arr = (float *)malloc(c * sizeof(float));
  float *sum_dy = (float *)calloc(c, sizeof(float));
  float *sum_dy_xhat = (float *)calloc(c, sizeof(float));
  if (!inv_std_arr || !sum_dy || !sum_dy_xhat) {
    free(inv_std_arr);
    free(sum_dy);
    free(sum_dy_xhat);
    log_error("Error: st_batchnorm2d_backward allocation failed.");
    return false;
  }

  for (size_t ci = 0; ci < c; ++ci) {
    inv_std_arr[ci] = 1.0f / sqrtf(var->values[ci] + epsilon);
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
      const float inv_std = inv_std_arr[ci];
      const float g = gamma ? gamma->values[ci] : 1.0f;

      float local_sum_dy = 0.0f;
      float local_sum_dy_xhat = 0.0f;
      float local_grad_gamma = 0.0f;
      float local_grad_beta = 0.0f;

      for (size_t i = 0; i < spatial; ++i) {
        const float dy = go_plane[i];
        const float x_hat = (in_plane[i] - mu) * inv_std;

        local_sum_dy += dy * g;
        local_sum_dy_xhat += dy * g * x_hat;
        local_grad_gamma += dy * x_hat;
        local_grad_beta += dy;
      }

      sum_dy[ci] += local_sum_dy;
      sum_dy_xhat[ci] += local_sum_dy_xhat;
      if (grad_gamma) {
        grad_gamma->values[ci] += local_grad_gamma;
      }
      if (grad_beta) {
        grad_beta->values[ci] += local_grad_beta;
      }
    }
  }

  /* Pass 2: compute grad_input — parallelized over N*C planes.
   * dx_i = inv_std / m * (m * dy_i*gamma - sum_dy - x_hat_i * sum_dy_xhat) */
  const float inv_m = 1.0f / (float)m;
  const size_t nc = n * c;

#pragma omp parallel for schedule(static) if (nc > 4)
  for (size_t nci = 0; nci < nc; ++nci) {
    const size_t ni = nci / c;
    const size_t ci = nci % c;
    const float *go_plane = grad_output->values + (ni * c + ci) * spatial;
    const float *in_plane = input->values + (ni * c + ci) * spatial;
    float *gi_plane = grad_input->values + (ni * c + ci) * spatial;
    const float mu = mean->values[ci];
    const float inv_std = inv_std_arr[ci];
    const float g = gamma ? gamma->values[ci] : 1.0f;
    const float sd = sum_dy[ci];
    const float sdx = sum_dy_xhat[ci];
    const float scale = inv_std * inv_m;

    for (size_t i = 0; i < spatial; ++i) {
      const float x_hat = (in_plane[i] - mu) * inv_std;
      gi_plane[i] = scale * ((float)m * go_plane[i] * g - sd - x_hat * sdx);
    }
  }

  free(inv_std_arr);
  free(sum_dy);
  free(sum_dy_xhat);
  return true;
}

/* ---- Fused Batch Normalization 2D + ReLU Forward ---- */

bool st_batchnorm2d_forward_relu(const FloatTensor *input,
                                 const FloatTensor *gamma,
                                 const FloatTensor *beta, float epsilon,
                                 FloatTensor *output, FloatTensor *mean,
                                 FloatTensor *var) {
  if (!st_bn_is_valid_4d(input) || !st_bn_is_valid_4d(output)) {
    log_error(
        "Error: st_batchnorm2d_forward_relu expects valid contiguous 4D "
        "tensors.");
    return false;
  }

  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];
  const size_t spatial = h * w;
  const size_t m = n * spatial;

  if (output->shape[0] != n || output->shape[1] != c ||
      output->shape[2] != h || output->shape[3] != w) {
    log_error("Error: st_batchnorm2d_forward_relu output shape mismatch.");
    return false;
  }

  if (!st_bn_is_valid_1d(mean, c) || !st_bn_is_valid_1d(var, c)) {
    log_error(
        "Error: st_batchnorm2d_forward_relu mean/var must be contiguous [C].");
    return false;
  }

  if (gamma && !st_bn_is_valid_1d(gamma, c)) {
    log_error(
        "Error: st_batchnorm2d_forward_relu gamma must be contiguous [C].");
    return false;
  }
  if (beta && !st_bn_is_valid_1d(beta, c)) {
    log_error(
        "Error: st_batchnorm2d_forward_relu beta must be contiguous [C].");
    return false;
  }

  const float inv_m = 1.0f / (float)m;

  /* Step 1: Compute per-channel mean. */
  memset(mean->values, 0, c * sizeof(float));
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *plane = input->values + (ni * c + ci) * spatial;
#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
      float plane_sum = 0.0f;
      vDSP_sve(plane, 1, &plane_sum, (vDSP_Length)spatial);
      mean->values[ci] += plane_sum;
#else
      float sum = 0.0f;
      for (size_t i = 0; i < spatial; ++i) {
        sum += plane[i];
      }
      mean->values[ci] += sum;
#endif
    }
  }
  for (size_t ci = 0; ci < c; ++ci) {
    mean->values[ci] *= inv_m;
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
    var->values[ci] *= inv_m;
  }

  /* Pre-compute inv_std per channel. */
  float *inv_std = (float *)malloc(c * sizeof(float));
  if (!inv_std) {
    log_error("Error: st_batchnorm2d_forward_relu allocation failed.");
    return false;
  }
  for (size_t ci = 0; ci < c; ++ci) {
    inv_std[ci] = 1.0f / sqrtf(var->values[ci] + epsilon);
  }

  /* Step 3: Normalize, scale, shift + ReLU — single fused pass. */
  const size_t nc = n * c;

#pragma omp parallel for schedule(static) if (nc > 4)
  for (size_t nci = 0; nci < nc; ++nci) {
    const size_t ni = nci / c;
    const size_t ci = nci % c;
    const float *in_plane = input->values + (ni * c + ci) * spatial;
    float *out_plane = output->values + (ni * c + ci) * spatial;
    const float mu = mean->values[ci];
    const float is = inv_std[ci];
    const float g = gamma ? gamma->values[ci] : 1.0f;
    const float b = beta ? beta->values[ci] : 0.0f;
    const float g_is = g * is;

    for (size_t i = 0; i < spatial; ++i) {
      const float bn_val = (in_plane[i] - mu) * g_is + b;
      out_plane[i] = fmaxf(0.0f, bn_val);
    }
  }

  free(inv_std);
  return true;
}

/* ---- Fused Batch Normalization 2D + ReLU Backward ---- */

bool st_batchnorm2d_backward_relu(const FloatTensor *grad_output,
                                  const FloatTensor *input,
                                  const FloatTensor *bn_output,
                                  const FloatTensor *mean,
                                  const FloatTensor *var,
                                  const FloatTensor *gamma, float epsilon,
                                  FloatTensor *grad_input,
                                  FloatTensor *grad_gamma,
                                  FloatTensor *grad_beta) {
  if (!st_bn_is_valid_4d(grad_output) || !st_bn_is_valid_4d(input) ||
      !st_bn_is_valid_4d(bn_output) || !st_bn_is_valid_4d(grad_input)) {
    log_error(
        "Error: st_batchnorm2d_backward_relu expects valid contiguous 4D "
        "tensors.");
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
        "Error: st_batchnorm2d_backward_relu mean/var must be contiguous [C].");
    return false;
  }

  if (gamma && !st_bn_is_valid_1d(gamma, c)) {
    log_error(
        "Error: st_batchnorm2d_backward_relu gamma must be contiguous [C].");
    return false;
  }

  if (grad_gamma && !st_bn_is_valid_1d(grad_gamma, c)) {
    log_error(
        "Error: st_batchnorm2d_backward_relu grad_gamma must be contiguous "
        "[C].");
    return false;
  }
  if (grad_beta && !st_bn_is_valid_1d(grad_beta, c)) {
    log_error(
        "Error: st_batchnorm2d_backward_relu grad_beta must be contiguous "
        "[C].");
    return false;
  }

  /* Pre-compute inv_std per channel. */
  float *inv_std_arr = (float *)malloc(c * sizeof(float));
  float *sum_dy = (float *)calloc(c, sizeof(float));
  float *sum_dy_xhat = (float *)calloc(c, sizeof(float));
  if (!inv_std_arr || !sum_dy || !sum_dy_xhat) {
    free(inv_std_arr);
    free(sum_dy);
    free(sum_dy_xhat);
    log_error("Error: st_batchnorm2d_backward_relu allocation failed.");
    return false;
  }

  for (size_t ci = 0; ci < c; ++ci) {
    inv_std_arr[ci] = 1.0f / sqrtf(var->values[ci] + epsilon);
  }

  /* Pass 1: accumulate sum(dy) and sum(dy * x_hat) per channel,
   * and gradients w.r.t. gamma and beta.
   * Fused ReLU mask: dy_masked = (bn_output > 0) ? dy : 0 */
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
      const float *bo_plane = bn_output->values + (ni * c + ci) * spatial;
      const float mu = mean->values[ci];
      const float inv_std = inv_std_arr[ci];
      const float g = gamma ? gamma->values[ci] : 1.0f;

      float local_sum_dy = 0.0f;
      float local_sum_dy_xhat = 0.0f;
      float local_grad_gamma = 0.0f;
      float local_grad_beta = 0.0f;

      for (size_t i = 0; i < spatial; ++i) {
        /* Fused ReLU mask: zero gradient where bn_output <= 0 */
        const float relu_mask = (bo_plane[i] > 0.0f) ? 1.0f : 0.0f;
        const float dy = go_plane[i] * relu_mask;
        const float x_hat = (in_plane[i] - mu) * inv_std;

        local_sum_dy += dy * g;
        local_sum_dy_xhat += dy * g * x_hat;
        local_grad_gamma += dy * x_hat;
        local_grad_beta += dy;
      }

      sum_dy[ci] += local_sum_dy;
      sum_dy_xhat[ci] += local_sum_dy_xhat;
      if (grad_gamma) {
        grad_gamma->values[ci] += local_grad_gamma;
      }
      if (grad_beta) {
        grad_beta->values[ci] += local_grad_beta;
      }
    }
  }

  /* Pass 2: compute grad_input — fused with ReLU mask.
   * dx_i = relu_mask * inv_std / m * (m * dy_i*gamma - sum_dy -
   *         x_hat_i * sum_dy_xhat) */
  const float inv_m = 1.0f / (float)m;
  const size_t nc = n * c;

#pragma omp parallel for schedule(static) if (nc > 4)
  for (size_t nci = 0; nci < nc; ++nci) {
    const size_t ni = nci / c;
    const size_t ci = nci % c;
    const float *go_plane = grad_output->values + (ni * c + ci) * spatial;
    const float *in_plane = input->values + (ni * c + ci) * spatial;
    const float *bo_plane = bn_output->values + (ni * c + ci) * spatial;
    float *gi_plane = grad_input->values + (ni * c + ci) * spatial;
    const float mu = mean->values[ci];
    const float inv_std = inv_std_arr[ci];
    const float g = gamma ? gamma->values[ci] : 1.0f;
    const float sd = sum_dy[ci];
    const float sdx = sum_dy_xhat[ci];
    const float scale = inv_std * inv_m;

    for (size_t i = 0; i < spatial; ++i) {
      const float relu_mask = (bo_plane[i] > 0.0f) ? 1.0f : 0.0f;
      const float dy = go_plane[i] * relu_mask;
      const float x_hat = (in_plane[i] - mu) * inv_std;
      gi_plane[i] = scale * ((float)m * dy * g - sd - x_hat * sdx);
    }
  }

  free(inv_std_arr);
  free(sum_dy);
  free(sum_dy_xhat);
  return true;
}
