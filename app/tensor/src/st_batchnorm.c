/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_batchnorm.h"
#include "st_backend.h"
#include "st_buffer.h"
#include "st_dtype.h"

#include <log.h>
#include <math.h>

#if defined(USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
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

static float st_bn_sum_plane(const float *plane, size_t spatial) {
#if defined(USE_ACCELERATE)
  float plane_sum = 0.0f;
  vDSP_sve(plane, 1, &plane_sum, (vDSP_Length)spatial);
  return plane_sum;
#else
  float sum = 0.0f;
  for (size_t i = 0; i < spatial; ++i) {
    sum += plane[i];
  }
  return sum;
#endif
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

  /* ---- bf16 promotion: convert inputs to f32, compute, convert back ---- */
  const bool need_bf16 =
      (input->dtype == ST_DTYPE_BF16) ||
      (output->dtype == ST_DTYPE_BF16) ||
      (gamma && gamma->dtype == ST_DTYPE_BF16) ||
      (beta && beta->dtype == ST_DTYPE_BF16);

  if (need_bf16) {
    FloatTensor *in_f32 = (input->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(input)
                              : (FloatTensor *)input;
    FloatTensor *g_f32 = NULL;
    if (gamma) {
      g_f32 = (gamma->dtype == ST_DTYPE_BF16) ? st_to_f32(gamma)
                                               : (FloatTensor *)gamma;
    }
    FloatTensor *b_f32 = NULL;
    if (beta) {
      b_f32 = (beta->dtype == ST_DTYPE_BF16) ? st_to_f32(beta)
                                              : (FloatTensor *)beta;
    }
    FloatTensor *out_f32 = st_create(output->ndim, output->shape);
    if (!in_f32 || !out_f32 || (gamma && !g_f32) || (beta && !b_f32)) {
      if (in_f32 != input) st_destroy(in_f32);
      if (g_f32 && g_f32 != gamma) st_destroy(g_f32);
      if (b_f32 && b_f32 != beta) st_destroy(b_f32);
      st_destroy(out_f32);
      log_error(
          "Error: st_batchnorm2d_forward bf16 promotion allocation failed.");
      return false;
    }

    bool ok =
        st_batchnorm2d_forward(in_f32, g_f32, b_f32, epsilon, out_f32, mean,
                               var);

    if (ok && output->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(out_f32->values, (uint16_t *)output->values,
                          output->numel);
    } else if (ok) {
      memcpy(output->values, out_f32->values, output->numel * sizeof(float));
    }

    if (in_f32 != input) st_destroy(in_f32);
    if (g_f32 && g_f32 != gamma) st_destroy(g_f32);
    if (b_f32 && b_f32 != beta) st_destroy(b_f32);
    st_destroy(out_f32);
    return ok;
  }

  /* ---- MPS dispatch via backend vtable ---- */
  {
    const StBackend *be = st_select_backend(ST_OP_BATCHNORM2D_FORWARD, input);
    if (be && be->batchnorm2d_forward) {
      bool ok = be->batchnorm2d_forward(input, gamma, beta, epsilon, output,
                                        mean, var);
      if (ok) return true;
      /* Fallback to CPU on MPS failure. */
    }
  }

  const float inv_m = 1.0f / (float)m;

#pragma omp parallel for schedule(static) if (c > 1)
  for (size_t ci = 0; ci < c; ++ci) {
    float sum = 0.0f;
    for (size_t ni = 0; ni < n; ++ni) {
      const float *plane = input->values + (ni * c + ci) * spatial;
      sum += st_bn_sum_plane(plane, spatial);
    }

    const float mu = sum * inv_m;
    mean->values[ci] = mu;

    float sum_sq = 0.0f;
    for (size_t ni = 0; ni < n; ++ni) {
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      for (size_t i = 0; i < spatial; ++i) {
        const float diff = in_plane[i] - mu;
        sum_sq += diff * diff;
      }
    }

    const float variance = sum_sq * inv_m;
    var->values[ci] = variance;

    const float inv_std = 1.0f / sqrtf(variance + epsilon);
    const float g = gamma ? gamma->values[ci] : 1.0f;
    const float b = beta ? beta->values[ci] : 0.0f;
    const float g_is = g * inv_std;

    for (size_t ni = 0; ni < n; ++ni) {
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      float *out_plane = output->values + (ni * c + ci) * spatial;
      for (size_t i = 0; i < spatial; ++i) {
        out_plane[i] = (in_plane[i] - mu) * g_is + b;
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

  /* ---- bf16 promotion ---- */
  const bool need_bf16_bw =
      (grad_output->dtype == ST_DTYPE_BF16) ||
      (input->dtype == ST_DTYPE_BF16) ||
      (grad_input->dtype == ST_DTYPE_BF16) ||
      (gamma && gamma->dtype == ST_DTYPE_BF16) ||
      (grad_gamma && grad_gamma->dtype == ST_DTYPE_BF16) ||
      (grad_beta && grad_beta->dtype == ST_DTYPE_BF16);

  if (need_bf16_bw) {
    FloatTensor *go_f32 = (grad_output->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(grad_output)
                              : (FloatTensor *)grad_output;
    FloatTensor *in_f32 = (input->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(input)
                              : (FloatTensor *)input;
    FloatTensor *g_f32 = NULL;
    if (gamma) {
      g_f32 = (gamma->dtype == ST_DTYPE_BF16) ? st_to_f32(gamma)
                                               : (FloatTensor *)gamma;
    }
    FloatTensor *gi_f32 = st_create(grad_input->ndim, grad_input->shape);
    FloatTensor *gg_f32 = grad_gamma ? st_create(1, grad_gamma->shape) : NULL;
    FloatTensor *gb_f32 = grad_beta ? st_create(1, grad_beta->shape) : NULL;

    if (!go_f32 || !in_f32 || !gi_f32 || (gamma && !g_f32) ||
        (grad_gamma && !gg_f32) || (grad_beta && !gb_f32)) {
      if (go_f32 != grad_output) st_destroy(go_f32);
      if (in_f32 != input) st_destroy(in_f32);
      if (g_f32 && g_f32 != gamma) st_destroy(g_f32);
      st_destroy(gi_f32);
      st_destroy(gg_f32);
      st_destroy(gb_f32);
      log_error(
          "Error: st_batchnorm2d_backward bf16 promotion allocation failed.");
      return false;
    }

    bool ok = st_batchnorm2d_backward(go_f32, in_f32, mean, var, g_f32,
                                      epsilon, gi_f32, gg_f32, gb_f32);

    if (ok && grad_input->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(gi_f32->values, (uint16_t *)grad_input->values,
                          grad_input->numel);
    } else if (ok) {
      memcpy(grad_input->values, gi_f32->values,
             grad_input->numel * sizeof(float));
    }
    if (ok && grad_gamma) {
      if (grad_gamma->dtype == ST_DTYPE_BF16) {
        st_f32_to_bf16_bulk(gg_f32->values, (uint16_t *)grad_gamma->values,
                            grad_gamma->numel);
      } else {
        memcpy(grad_gamma->values, gg_f32->values,
               grad_gamma->numel * sizeof(float));
      }
    }
    if (ok && grad_beta) {
      if (grad_beta->dtype == ST_DTYPE_BF16) {
        st_f32_to_bf16_bulk(gb_f32->values, (uint16_t *)grad_beta->values,
                            grad_beta->numel);
      } else {
        memcpy(grad_beta->values, gb_f32->values,
               grad_beta->numel * sizeof(float));
      }
    }

    if (go_f32 != grad_output) st_destroy(go_f32);
    if (in_f32 != input) st_destroy(in_f32);
    if (g_f32 && g_f32 != gamma) st_destroy(g_f32);
    st_destroy(gi_f32);
    st_destroy(gg_f32);
    st_destroy(gb_f32);
    return ok;
  }

  const float inv_m = 1.0f / (float)m;

#pragma omp parallel for schedule(static) if (c > 1)
  for (size_t ci = 0; ci < c; ++ci) {
    const float mu = mean->values[ci];
    const float inv_std = 1.0f / sqrtf(var->values[ci] + epsilon);
    const float g = gamma ? gamma->values[ci] : 1.0f;

    float channel_sum_dy = 0.0f;
    float channel_sum_dy_xhat = 0.0f;
    float channel_grad_gamma = 0.0f;
    float channel_grad_beta = 0.0f;

    for (size_t ni = 0; ni < n; ++ni) {
      const float *go_plane = grad_output->values + (ni * c + ci) * spatial;
      const float *in_plane = input->values + (ni * c + ci) * spatial;

      for (size_t i = 0; i < spatial; ++i) {
        const float dy = go_plane[i];
        const float x_hat = (in_plane[i] - mu) * inv_std;
        const float scaled_dy = dy * g;

        channel_sum_dy += scaled_dy;
        channel_sum_dy_xhat += scaled_dy * x_hat;
        channel_grad_gamma += dy * x_hat;
        channel_grad_beta += dy;
      }
    }

    if (grad_gamma) {
      grad_gamma->values[ci] = channel_grad_gamma;
    }
    if (grad_beta) {
      grad_beta->values[ci] = channel_grad_beta;
    }

    const float scale = inv_std * inv_m;
    for (size_t ni = 0; ni < n; ++ni) {
      const float *go_plane = grad_output->values + (ni * c + ci) * spatial;
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      float *gi_plane = grad_input->values + (ni * c + ci) * spatial;

      for (size_t i = 0; i < spatial; ++i) {
        const float x_hat = (in_plane[i] - mu) * inv_std;
        gi_plane[i] =
            scale * ((float)m * go_plane[i] * g - channel_sum_dy -
                     x_hat * channel_sum_dy_xhat);
      }
    }
  }

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

  /* ---- bf16 promotion ---- */
  const bool need_bf16_promotion =
      (input->dtype == ST_DTYPE_BF16) ||
      (output->dtype == ST_DTYPE_BF16) ||
      (gamma && gamma->dtype == ST_DTYPE_BF16) ||
      (beta && beta->dtype == ST_DTYPE_BF16);

  if (need_bf16_promotion) {
    FloatTensor *in_f32 = (input->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(input)
                              : (FloatTensor *)input;
    FloatTensor *g_f32 = NULL;
    if (gamma) {
      g_f32 = (gamma->dtype == ST_DTYPE_BF16) ? st_to_f32(gamma)
                                               : (FloatTensor *)gamma;
    }
    FloatTensor *b_f32 = NULL;
    if (beta) {
      b_f32 = (beta->dtype == ST_DTYPE_BF16) ? st_to_f32(beta)
                                              : (FloatTensor *)beta;
    }
    FloatTensor *out_f32 = st_create(output->ndim, output->shape);
    if (!in_f32 || !out_f32 || (gamma && !g_f32) || (beta && !b_f32)) {
      if (in_f32 != input) st_destroy(in_f32);
      if (g_f32 && g_f32 != gamma) st_destroy(g_f32);
      if (b_f32 && b_f32 != beta) st_destroy(b_f32);
      st_destroy(out_f32);
      log_error(
          "Error: st_batchnorm2d_forward_relu bf16 promotion allocation "
          "failed.");
      return false;
    }

    bool ok = st_batchnorm2d_forward_relu(in_f32, g_f32, b_f32, epsilon,
                                          out_f32, mean, var);

    if (ok && output->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(out_f32->values, (uint16_t *)output->values,
                          output->numel);
    } else if (ok) {
      memcpy(output->values, out_f32->values, output->numel * sizeof(float));
    }

    if (in_f32 != input) st_destroy(in_f32);
    if (g_f32 && g_f32 != gamma) st_destroy(g_f32);
    if (b_f32 && b_f32 != beta) st_destroy(b_f32);
    st_destroy(out_f32);
    return ok;
  }

  const float inv_m = 1.0f / (float)m;

#pragma omp parallel for schedule(static) if (c > 1)
  for (size_t ci = 0; ci < c; ++ci) {
    float sum = 0.0f;
    for (size_t ni = 0; ni < n; ++ni) {
      const float *plane = input->values + (ni * c + ci) * spatial;
      sum += st_bn_sum_plane(plane, spatial);
    }

    const float mu = sum * inv_m;
    mean->values[ci] = mu;

    float sum_sq = 0.0f;
    for (size_t ni = 0; ni < n; ++ni) {
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      for (size_t i = 0; i < spatial; ++i) {
        const float diff = in_plane[i] - mu;
        sum_sq += diff * diff;
      }
    }

    const float variance = sum_sq * inv_m;
    var->values[ci] = variance;

    const float inv_std = 1.0f / sqrtf(variance + epsilon);
    const float g = gamma ? gamma->values[ci] : 1.0f;
    const float b = beta ? beta->values[ci] : 0.0f;
    const float g_is = g * inv_std;

    for (size_t ni = 0; ni < n; ++ni) {
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      float *out_plane = output->values + (ni * c + ci) * spatial;
      for (size_t i = 0; i < spatial; ++i) {
        const float bn_val = (in_plane[i] - mu) * g_is + b;
        out_plane[i] = fmaxf(0.0f, bn_val);
      }
    }
  }

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

  /* ---- bf16 promotion ---- */
  const bool need_bf16_br =
      (grad_output->dtype == ST_DTYPE_BF16) ||
      (input->dtype == ST_DTYPE_BF16) ||
      (bn_output->dtype == ST_DTYPE_BF16) ||
      (grad_input->dtype == ST_DTYPE_BF16) ||
      (gamma && gamma->dtype == ST_DTYPE_BF16) ||
      (grad_gamma && grad_gamma->dtype == ST_DTYPE_BF16) ||
      (grad_beta && grad_beta->dtype == ST_DTYPE_BF16);

  if (need_bf16_br) {
    FloatTensor *go_f32 = (grad_output->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(grad_output)
                              : (FloatTensor *)grad_output;
    FloatTensor *in_f32 = (input->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(input)
                              : (FloatTensor *)input;
    FloatTensor *bo_f32 = (bn_output->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(bn_output)
                              : (FloatTensor *)bn_output;
    FloatTensor *g_f32 = NULL;
    if (gamma) {
      g_f32 = (gamma->dtype == ST_DTYPE_BF16) ? st_to_f32(gamma)
                                               : (FloatTensor *)gamma;
    }
    FloatTensor *gi_f32 = st_create(grad_input->ndim, grad_input->shape);
    FloatTensor *gg_f32 = grad_gamma ? st_create(1, grad_gamma->shape) : NULL;
    FloatTensor *gb_f32 = grad_beta ? st_create(1, grad_beta->shape) : NULL;

    if (!go_f32 || !in_f32 || !bo_f32 || !gi_f32 || (gamma && !g_f32) ||
        (grad_gamma && !gg_f32) || (grad_beta && !gb_f32)) {
      if (go_f32 != grad_output) st_destroy(go_f32);
      if (in_f32 != input) st_destroy(in_f32);
      if (bo_f32 != bn_output) st_destroy(bo_f32);
      if (g_f32 && g_f32 != gamma) st_destroy(g_f32);
      st_destroy(gi_f32);
      st_destroy(gg_f32);
      st_destroy(gb_f32);
      log_error(
          "Error: st_batchnorm2d_backward_relu bf16 promotion allocation "
          "failed.");
      return false;
    }

    bool ok = st_batchnorm2d_backward_relu(go_f32, in_f32, bo_f32, mean, var,
                                           g_f32, epsilon, gi_f32, gg_f32,
                                           gb_f32);

    if (ok && grad_input->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(gi_f32->values, (uint16_t *)grad_input->values,
                          grad_input->numel);
    } else if (ok) {
      memcpy(grad_input->values, gi_f32->values,
             grad_input->numel * sizeof(float));
    }
    if (ok && grad_gamma) {
      if (grad_gamma->dtype == ST_DTYPE_BF16) {
        st_f32_to_bf16_bulk(gg_f32->values, (uint16_t *)grad_gamma->values,
                            grad_gamma->numel);
      } else {
        memcpy(grad_gamma->values, gg_f32->values,
               grad_gamma->numel * sizeof(float));
      }
    }
    if (ok && grad_beta) {
      if (grad_beta->dtype == ST_DTYPE_BF16) {
        st_f32_to_bf16_bulk(gb_f32->values, (uint16_t *)grad_beta->values,
                            grad_beta->numel);
      } else {
        memcpy(grad_beta->values, gb_f32->values,
               grad_beta->numel * sizeof(float));
      }
    }

    if (go_f32 != grad_output) st_destroy(go_f32);
    if (in_f32 != input) st_destroy(in_f32);
    if (bo_f32 != bn_output) st_destroy(bo_f32);
    if (g_f32 && g_f32 != gamma) st_destroy(g_f32);
    st_destroy(gi_f32);
    st_destroy(gg_f32);
    st_destroy(gb_f32);
    return ok;
  }

  const float inv_m = 1.0f / (float)m;

#pragma omp parallel for schedule(static) if (c > 1)
  for (size_t ci = 0; ci < c; ++ci) {
    const float mu = mean->values[ci];
    const float inv_std = 1.0f / sqrtf(var->values[ci] + epsilon);
    const float g = gamma ? gamma->values[ci] : 1.0f;

    float channel_sum_dy = 0.0f;
    float channel_sum_dy_xhat = 0.0f;
    float channel_grad_gamma = 0.0f;
    float channel_grad_beta = 0.0f;

    for (size_t ni = 0; ni < n; ++ni) {
      const float *go_plane = grad_output->values + (ni * c + ci) * spatial;
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      const float *bo_plane = bn_output->values + (ni * c + ci) * spatial;

      for (size_t i = 0; i < spatial; ++i) {
        const float relu_mask = (bo_plane[i] > 0.0f) ? 1.0f : 0.0f;
        const float dy = go_plane[i] * relu_mask;
        const float x_hat = (in_plane[i] - mu) * inv_std;
        const float scaled_dy = dy * g;

        channel_sum_dy += scaled_dy;
        channel_sum_dy_xhat += scaled_dy * x_hat;
        channel_grad_gamma += dy * x_hat;
        channel_grad_beta += dy;
      }
    }

    if (grad_gamma) {
      grad_gamma->values[ci] = channel_grad_gamma;
    }
    if (grad_beta) {
      grad_beta->values[ci] = channel_grad_beta;
    }

    const float scale = inv_std * inv_m;
    for (size_t ni = 0; ni < n; ++ni) {
      const float *go_plane = grad_output->values + (ni * c + ci) * spatial;
      const float *in_plane = input->values + (ni * c + ci) * spatial;
      const float *bo_plane = bn_output->values + (ni * c + ci) * spatial;
      float *gi_plane = grad_input->values + (ni * c + ci) * spatial;

      for (size_t i = 0; i < spatial; ++i) {
        const float relu_mask = (bo_plane[i] > 0.0f) ? 1.0f : 0.0f;
        const float dy = go_plane[i] * relu_mask;
        const float x_hat = (in_plane[i] - mu) * inv_std;
        gi_plane[i] =
            scale * ((float)m * dy * g - channel_sum_dy -
                     x_hat * channel_sum_dy_xhat);
      }
    }
  }

  return true;
}
