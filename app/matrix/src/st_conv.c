/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_conv.h"

#include <log.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const char *g_last_backend = "none";

static bool st_conv2d_is_valid_tensor(const FloatTensor *t, size_t ndim) {
  return t != NULL && t->values != NULL && t->ndim == ndim;
}

static bool st_conv2d_reference_nchw(const FloatTensor *input,
                                     const FloatTensor *weight,
                                     const FloatTensor *bias,
                                     const StConv2dParams *params,
                                     FloatTensor *output) {
  const size_t n = input->shape[0];
  const size_t c_in = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];

  const size_t c_out = weight->shape[0];
  const size_t k_h = weight->shape[2];
  const size_t k_w = weight->shape[3];

  const size_t out_h = output->shape[2];
  const size_t out_w = output->shape[3];

  const size_t stride_h = params->stride_h;
  const size_t stride_w = params->stride_w;
  const size_t pad_h = params->pad_h;
  const size_t pad_w = params->pad_w;
  const size_t dil_h = params->dilation_h;
  const size_t dil_w = params->dilation_w;

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t co = 0; co < c_out; ++co) {
      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
          float sum = 0.0f;

          for (size_t ci = 0; ci < c_in; ++ci) {
            for (size_t kh = 0; kh < k_h; ++kh) {
              for (size_t kw = 0; kw < k_w; ++kw) {
                const ptrdiff_t in_h =
                    (ptrdiff_t)(oh * stride_h + kh * dil_h) - (ptrdiff_t)pad_h;
                const ptrdiff_t in_w =
                    (ptrdiff_t)(ow * stride_w + kw * dil_w) - (ptrdiff_t)pad_w;

                if (in_h < 0 || in_w < 0 || (size_t)in_h >= h ||
                    (size_t)in_w >= w) {
                  continue;
                }

                const size_t in_idx =
                    ((ni * c_in + ci) * h + (size_t)in_h) * w + (size_t)in_w;
                const size_t w_idx =
                    ((co * c_in + ci) * k_h + kh) * k_w + kw;
                sum += input->values[in_idx] * weight->values[w_idx];
              }
            }
          }

          if (bias) {
            sum += bias->values[co];
          }

          const size_t out_idx = ((ni * c_out + co) * out_h + oh) * out_w + ow;
          output->values[out_idx] = sum;
        }
      }
    }
  }

  g_last_backend = "reference";
  return true;
}

static bool st_conv2d_cpu_opt_nchw(const FloatTensor *input,
                                   const FloatTensor *weight,
                                   const FloatTensor *bias,
                                   const StConv2dParams *params,
                                   FloatTensor *output) {
  const size_t n = input->shape[0];
  const size_t c_in = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];

  const size_t c_out = weight->shape[0];
  const size_t k_h = weight->shape[2];
  const size_t k_w = weight->shape[3];

  const size_t out_h = output->shape[2];
  const size_t out_w = output->shape[3];

  const size_t stride_h = params->stride_h;
  const size_t stride_w = params->stride_w;
  const size_t pad_h = params->pad_h;
  const size_t pad_w = params->pad_w;
  const size_t dil_h = params->dilation_h;
  const size_t dil_w = params->dilation_w;

#pragma omp parallel for collapse(2) if (n * c_out >= 4)
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t co = 0; co < c_out; ++co) {
      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
          float sum = 0.0f;
          const ptrdiff_t in_w_base_common =
              (ptrdiff_t)(ow * stride_w) - (ptrdiff_t)pad_w;
          const bool can_use_fast_kw =
              (dil_w == 1) &&
              (in_w_base_common >= 0) &&
              ((size_t)in_w_base_common + k_w <= w);

          for (size_t ci = 0; ci < c_in; ++ci) {
            for (size_t kh = 0; kh < k_h; ++kh) {
              const ptrdiff_t in_h_base =
                  (ptrdiff_t)(oh * stride_h + kh * dil_h) - (ptrdiff_t)pad_h;
              if (in_h_base < 0 || (size_t)in_h_base >= h) {
                continue;
              }

              const size_t in_row_off =
                  ((ni * c_in + ci) * h + (size_t)in_h_base) * w;
              const size_t w_off = ((co * c_in + ci) * k_h + kh) * k_w;

              if (can_use_fast_kw) {
                const float *in_ptr =
                    &input->values[in_row_off + (size_t)in_w_base_common];
                const float *w_ptr = &weight->values[w_off];

#ifdef __ARM_NEON
                float32x4_t vacc = vdupq_n_f32(0.0f);
                size_t kw = 0;
                for (; kw + 4 <= k_w; kw += 4) {
                  const float32x4_t vin = vld1q_f32(in_ptr + kw);
                  const float32x4_t vw = vld1q_f32(w_ptr + kw);
                  vacc = vmlaq_f32(vacc, vin, vw);
                }
                sum += vaddvq_f32(vacc);
                for (; kw < k_w; ++kw) {
                  sum += in_ptr[kw] * w_ptr[kw];
                }
#else
                for (size_t kw = 0; kw < k_w; ++kw) {
                  sum += in_ptr[kw] * w_ptr[kw];
                }
#endif
              } else {
                for (size_t kw = 0; kw < k_w; ++kw) {
                  const ptrdiff_t in_w_cur =
                      in_w_base_common + (ptrdiff_t)(kw * dil_w);

                  if (in_w_cur < 0 || (size_t)in_w_cur >= w) {
                    continue;
                  }

                  const size_t in_idx = in_row_off + (size_t)in_w_cur;
                  const size_t w_idx = w_off + kw;
                  sum += input->values[in_idx] * weight->values[w_idx];
                }
              }
            }
          }

          if (bias) {
            sum += bias->values[co];
          }

          const size_t out_idx = ((ni * c_out + co) * out_h + oh) * out_w + ow;
          output->values[out_idx] = sum;
        }
      }
    }
  }

  g_last_backend = "cpu_opt";
  return true;
}

static bool st_conv2d_should_use_cpu_opt(size_t n, size_t c_in, size_t c_out,
                                         size_t out_h, size_t out_w,
                                         size_t k_h, size_t k_w) {
  const size_t out_elems = n * c_out * out_h * out_w;
  const size_t kernel_volume = c_in * k_h * k_w;
  return out_elems >= 16384 && kernel_volume >= 27;
}

static bool st_conv2d_bnns_nchw(const FloatTensor *input,
                                const FloatTensor *weight,
                                const FloatTensor *bias,
                                const StConv2dParams *params,
                                FloatTensor *output) {
#if defined(USE_ACCELERATE) && defined(__APPLE__)
  // Placeholder for initial BNNS integration point.
  // In this first step we keep behavior correct and stable,
  // while the actual BNNS descriptor wiring follows in next iteration.
  (void)input;
  (void)weight;
  (void)bias;
  (void)params;
  (void)output;
  return false;
#else
  (void)input;
  (void)weight;
  (void)bias;
  (void)params;
  (void)output;
  return false;
#endif
}

StConv2dParams st_conv2d_default_params(void) {
  StConv2dParams p;
  p.stride_h = 1;
  p.stride_w = 1;
  p.pad_h = 0;
  p.pad_w = 0;
  p.dilation_h = 1;
  p.dilation_w = 1;
  p.backend = ST_CONV_BACKEND_AUTO;
  return p;
}

bool st_conv2d_output_hw(size_t in_h, size_t in_w, size_t kernel_h,
                         size_t kernel_w, const StConv2dParams *params,
                         size_t *out_h, size_t *out_w) {
  if (!params || !out_h || !out_w || params->stride_h == 0 ||
      params->stride_w == 0 || params->dilation_h == 0 ||
      params->dilation_w == 0 || kernel_h == 0 || kernel_w == 0) {
    return false;
  }

  const size_t eff_h = params->dilation_h * (kernel_h - 1) + 1;
  const size_t eff_w = params->dilation_w * (kernel_w - 1) + 1;

  if (in_h + 2 * params->pad_h < eff_h || in_w + 2 * params->pad_w < eff_w) {
    return false;
  }

  *out_h = (in_h + 2 * params->pad_h - eff_h) / params->stride_h + 1;
  *out_w = (in_w + 2 * params->pad_w - eff_w) / params->stride_w + 1;
  return true;
}

bool st_conv2d_nchw(const FloatTensor *input, const FloatTensor *weight,
                    const FloatTensor *bias, const StConv2dParams *params,
                    FloatTensor *output) {
  StConv2dParams local = st_conv2d_default_params();
  if (params) {
    local = *params;
  }

  if (!st_conv2d_is_valid_tensor(input, 4) ||
      !st_conv2d_is_valid_tensor(weight, 4) ||
      !st_conv2d_is_valid_tensor(output, 4)) {
    log_error("Error: st_conv2d_nchw expects valid 4D tensors.");
    return false;
  }

  if (!st_is_contiguous(input) || !st_is_contiguous(weight) ||
      !st_is_contiguous(output)) {
    log_error("Error: st_conv2d_nchw currently requires contiguous tensors.");
    return false;
  }

  const size_t n = input->shape[0];
  const size_t c_in = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];

  const size_t c_out = weight->shape[0];
  const size_t w_c_in = weight->shape[1];
  const size_t k_h = weight->shape[2];
  const size_t k_w = weight->shape[3];

  if (c_in != w_c_in) {
    log_error("Error: st_conv2d_nchw channel mismatch.");
    return false;
  }

  if (bias) {
    if (!st_conv2d_is_valid_tensor(bias, 1) || bias->shape[0] != c_out ||
        !st_is_contiguous(bias)) {
      log_error("Error: st_conv2d_nchw bias must be contiguous [C_out].");
      return false;
    }
  }

  size_t out_h = 0;
  size_t out_w = 0;
  if (!st_conv2d_output_hw(h, w, k_h, k_w, &local, &out_h, &out_w)) {
    log_error("Error: st_conv2d_nchw invalid output shape parameters.");
    return false;
  }

  if (output->shape[0] != n || output->shape[1] != c_out ||
      output->shape[2] != out_h || output->shape[3] != out_w) {
    log_error("Error: st_conv2d_nchw output tensor shape mismatch.");
    return false;
  }

  bool ok = false;
  switch (local.backend) {
    case ST_CONV_BACKEND_REFERENCE:
      ok = st_conv2d_reference_nchw(input, weight, bias, &local, output);
      if (!ok) {
        log_error("Error: st_conv2d_nchw reference path failed.");
        return false;
      }
      return true;

    case ST_CONV_BACKEND_CPU_OPT:
      ok = st_conv2d_cpu_opt_nchw(input, weight, bias, &local, output);
      if (!ok) {
        log_error("Error: st_conv2d_nchw cpu_opt path failed.");
        return false;
      }
      return true;

    case ST_CONV_BACKEND_BNNS:
      ok = st_conv2d_bnns_nchw(input, weight, bias, &local, output);
      if (ok) {
        g_last_backend = "bnns";
        return true;
      }
      ok = st_conv2d_reference_nchw(input, weight, bias, &local, output);
      if (!ok) {
        log_error("Error: st_conv2d_nchw reference fallback path failed.");
        return false;
      }
      g_last_backend = "bnns_fallback_reference";
      return true;

    case ST_CONV_BACKEND_AUTO:
    default:
      ok = st_conv2d_bnns_nchw(input, weight, bias, &local, output);
      if (ok) {
        g_last_backend = "bnns";
        return true;
      }

      if (st_conv2d_should_use_cpu_opt(n, c_in, c_out, out_h, out_w, k_h,
                                       k_w)) {
        ok = st_conv2d_cpu_opt_nchw(input, weight, bias, &local, output);
        if (ok) {
          return true;
        }
      }

      ok = st_conv2d_reference_nchw(input, weight, bias, &local, output);
      if (!ok) {
        log_error("Error: st_conv2d_nchw reference path failed.");
        return false;
      }
      return true;
  }
}

const char *st_conv2d_last_backend(void) { return g_last_backend; }
