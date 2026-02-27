/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_conv.h"
#include "sm.h"

#include <log.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Thread-local: each thread records its own last-used backend. */
static _Thread_local const char *g_last_backend = "none";

static const double ST_CONV_MPS_MACS_THRESHOLD_DEFAULT = 2.0e8;
static const size_t ST_CONV_MPS_OUT_ELEMS_THRESHOLD_DEFAULT = 1000000u;

static double g_mps_macs_threshold = ST_CONV_MPS_MACS_THRESHOLD_DEFAULT;
static size_t g_mps_out_elems_threshold =
    ST_CONV_MPS_OUT_ELEMS_THRESHOLD_DEFAULT;
/* Atomic flag: safe concurrent read of the "already initialised" state. */
static atomic_bool g_mps_thresholds_initialized = false;

bool st_conv_set_mps_thresholds(double macs_threshold,
                size_t out_elems_threshold);
void st_conv_get_mps_thresholds(double *out_macs_threshold,
                size_t *out_out_elems_threshold);
void st_conv_reload_mps_thresholds_from_env(void);

static bool st_parse_positive_double(const char *text, double *out_value) {
  if (text == NULL || out_value == NULL) {
    return false;
  }

  errno = 0;
  char *end = NULL;
  const double value = strtod(text, &end);
  if (errno != 0 || end == text || (end != NULL && *end != '\0')) {
    return false;
  }
  if (!isfinite(value) || value <= 0.0) {
    return false;
  }

  *out_value = value;
  return true;
}

static bool st_parse_positive_size_t(const char *text, size_t *out_value) {
  if (text == NULL || out_value == NULL) {
    return false;
  }

  errno = 0;
  char *end = NULL;
  const unsigned long long value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || (end != NULL && *end != '\0')) {
    return false;
  }
  if (value == 0ull || value > (unsigned long long)SIZE_MAX) {
    return false;
  }

  *out_value = (size_t)value;
  return true;
}

static void st_conv_init_mps_thresholds_once(void) {
  if (atomic_load_explicit(&g_mps_thresholds_initialized,
                          memory_order_acquire)) {
    return;
  }
  st_conv_reload_mps_thresholds_from_env();
}

/* ---- Overflow-safe arithmetic helpers ---- */

/** Multiply two size_t values; return false on overflow. */
static inline bool st_safe_mul(size_t a, size_t b, size_t *out) {
  if (a != 0 && b > SIZE_MAX / a) {
    return false;
  }
  *out = a * b;
  return true;
}

/** Multiply three size_t values; return false on overflow. */
static inline bool st_safe_mul3(size_t a, size_t b, size_t c, size_t *out) {
  size_t ab;
  if (!st_safe_mul(a, b, &ab)) {
    return false;
  }
  return st_safe_mul(ab, c, out);
}

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

  /* Pre-compute safe ow range where all kw taps land inside the input
   * (requires dil_w == 1).  Left/right borders need per-pixel checks. */
  size_t ow_safe_lo = 0;
  size_t ow_safe_hi = 0;
  if (dil_w == 1) {
    ow_safe_lo = (pad_w > 0) ? (pad_w + stride_w - 1) / stride_w : 0;
    if (w + pad_w >= k_w) {
      ow_safe_hi = (w + pad_w - k_w) / stride_w + 1;
      if (ow_safe_hi > out_w) {
        ow_safe_hi = out_w;
      }
    }
    if (ow_safe_lo > ow_safe_hi) {
      ow_safe_hi = ow_safe_lo;
    }
  }

#pragma omp parallel for collapse(3) if (n * c_out * out_h >= 4)
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t co = 0; co < c_out; ++co) {
      for (size_t oh = 0; oh < out_h; ++oh) {
        /* --- Left border (full bounds checks) --- */
        for (size_t ow = 0; ow < ow_safe_lo; ++ow) {
          float sum = 0.0f;
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
              for (size_t kw = 0; kw < k_w; ++kw) {
                const ptrdiff_t in_w_cur =
                    (ptrdiff_t)(ow * stride_w + kw * dil_w) - (ptrdiff_t)pad_w;
                if (in_w_cur < 0 || (size_t)in_w_cur >= w) {
                  continue;
                }
                sum += input->values[in_row_off + (size_t)in_w_cur] *
                       weight->values[w_off + kw];
              }
            }
          }
          if (bias) {
            sum += bias->values[co];
          }
          output->values[((ni * c_out + co) * out_h + oh) * out_w + ow] = sum;
        }

        /* --- Inner region (kw taps guaranteed in-bounds, dil_w==1) --- */
        for (size_t ow = ow_safe_lo; ow < ow_safe_hi; ++ow) {
          float sum = 0.0f;
          const ptrdiff_t in_w_base =
              (ptrdiff_t)(ow * stride_w) - (ptrdiff_t)pad_w;

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
              const float *in_ptr =
                  &input->values[in_row_off + (size_t)in_w_base];
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
            }
          }
          if (bias) {
            sum += bias->values[co];
          }
          output->values[((ni * c_out + co) * out_h + oh) * out_w + ow] = sum;
        }

        /* --- Right border (full bounds checks) --- */
        for (size_t ow = ow_safe_hi; ow < out_w; ++ow) {
          float sum = 0.0f;
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
              for (size_t kw = 0; kw < k_w; ++kw) {
                const ptrdiff_t in_w_cur =
                    (ptrdiff_t)(ow * stride_w + kw * dil_w) - (ptrdiff_t)pad_w;
                if (in_w_cur < 0 || (size_t)in_w_cur >= w) {
                  continue;
                }
                sum += input->values[in_row_off + (size_t)in_w_cur] *
                       weight->values[w_off + kw];
              }
            }
          }
          if (bias) {
            sum += bias->values[co];
          }
          output->values[((ni * c_out + co) * out_h + oh) * out_w + ow] = sum;
        }
      }
    }
  }

  g_last_backend = "cpu_opt";
  return true;
}

/* ---- 1x1 fast-path: direct GEMM without im2col ---- */

static inline bool st_conv2d_is_1x1_no_pad(const StConv2dParams *params,
                                            size_t k_h, size_t k_w) {
  return k_h == 1 && k_w == 1 && params->pad_h == 0 && params->pad_w == 0 &&
         params->dilation_h == 1 && params->dilation_w == 1 &&
         params->stride_h == 1 && params->stride_w == 1;
}

static bool st_conv2d_gemm_1x1_nchw(const FloatTensor *input,
                                     const FloatTensor *weight,
                                     const FloatTensor *bias,
                                     FloatTensor *output) {
  const size_t n = input->shape[0];
  const size_t c_in = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];
  const size_t c_out = weight->shape[0];
  const size_t spatial = h * w;

  /* Weight is [C_out, C_in, 1, 1] — row-major [C_out, C_in]. */
  FloatMatrix w_view = {
      .rows = c_out, .cols = c_in, .capacity = 0, .values = weight->values};

  for (size_t ni = 0; ni < n; ++ni) {
    /* input_n  : [C_in,  H*W]  row-major view */
    FloatMatrix in_view = {.rows = c_in,
                           .cols = spatial,
                           .capacity = 0,
                           .values = input->values + ni * c_in * spatial};
    /* output_n : [C_out, H*W]  row-major view — matches NCHW layout */
    FloatMatrix out_view = {.rows = c_out,
                            .cols = spatial,
                            .capacity = 0,
                            .values = output->values + ni * c_out * spatial};

    /* output_n = weight × input_n  (no im2col needed for 1x1) */
    if (!sm_gemm(&out_view, 1.0f, &w_view, SM_NO_TRANSPOSE, &in_view,
                 SM_NO_TRANSPOSE, 0.0f)) {
      return false;
    }

    if (bias) {
      for (size_t co = 0; co < c_out; ++co) {
        const float b = bias->values[co];
        float *row = output->values + ni * c_out * spatial + co * spatial;
        for (size_t i = 0; i < spatial; ++i) {
          row[i] += b;
        }
      }
    }
  }

  g_last_backend = "gemm_1x1";
  return true;
}

/* ---- General GEMM path (im2col) ---- */

static bool st_conv2d_gemm_nchw(const FloatTensor *input,
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

  /* 1x1 fast-path: skip im2col entirely. */
  if (st_conv2d_is_1x1_no_pad(params, k_h, k_w)) {
    return st_conv2d_gemm_1x1_nchw(input, weight, bias, output);
  }

  const size_t stride_h = params->stride_h;
  const size_t stride_w = params->stride_w;
  const size_t pad_h = params->pad_h;
  const size_t pad_w = params->pad_w;
  const size_t dil_h = params->dilation_h;
  const size_t dil_w = params->dilation_w;

  size_t patch_size = 0;
  size_t out_spatial = 0;
  if (!st_safe_mul3(c_in, k_h, k_w, &patch_size) ||
      !st_safe_mul(out_h, out_w, &out_spatial)) {
    log_error("Error: st_conv2d_gemm_nchw size overflow in im2col dims.");
    return false;
  }

  FloatMatrix *w_mat = sm_create(patch_size, c_out);
  FloatMatrix *col = sm_create(out_spatial, patch_size);
  FloatMatrix *y = sm_create(out_spatial, c_out);
  if (!w_mat || !col || !y) {
    sm_destroy(w_mat);
    sm_destroy(col);
    sm_destroy(y);
    return false;
  }

  for (size_t co = 0; co < c_out; ++co) {
    for (size_t ci = 0; ci < c_in; ++ci) {
      for (size_t kh = 0; kh < k_h; ++kh) {
        for (size_t kw = 0; kw < k_w; ++kw) {
          const size_t k_idx = ((co * c_in + ci) * k_h + kh) * k_w + kw;
          const size_t row = ((ci * k_h) + kh) * k_w + kw;
          w_mat->values[row * c_out + co] = weight->values[k_idx];
        }
      }
    }
  }

  for (size_t ni = 0; ni < n; ++ni) {
    /* im2col: build column matrix — parallelise over output rows. */
#pragma omp parallel for if (out_spatial >= 256)
    for (size_t out_row = 0; out_row < out_spatial; ++out_row) {
      const size_t oh = out_row / out_w;
      const size_t ow = out_row % out_w;

      for (size_t ci = 0; ci < c_in; ++ci) {
        for (size_t kh = 0; kh < k_h; ++kh) {
          for (size_t kw = 0; kw < k_w; ++kw) {
            const ptrdiff_t in_h =
                (ptrdiff_t)(oh * stride_h + kh * dil_h) - (ptrdiff_t)pad_h;
            const ptrdiff_t in_w =
                (ptrdiff_t)(ow * stride_w + kw * dil_w) - (ptrdiff_t)pad_w;
            const size_t col_idx = ((ci * k_h) + kh) * k_w + kw;

            float v = 0.0f;
            if (in_h >= 0 && in_w >= 0 && (size_t)in_h < h &&
                (size_t)in_w < w) {
              const size_t in_idx =
                  ((ni * c_in + ci) * h + (size_t)in_h) * w + (size_t)in_w;
              v = input->values[in_idx];
            }

            col->values[out_row * patch_size + col_idx] = v;
          }
        }
      }
    }

    if (!sm_gemm(y, 1.0f, col, SM_NO_TRANSPOSE, w_mat, SM_NO_TRANSPOSE,
                 0.0f)) {
      sm_destroy(w_mat);
      sm_destroy(col);
      sm_destroy(y);
      return false;
    }

    for (size_t oh = 0; oh < out_h; ++oh) {
      for (size_t ow = 0; ow < out_w; ++ow) {
        const size_t out_row = oh * out_w + ow;
        for (size_t co = 0; co < c_out; ++co) {
          float v = y->values[out_row * c_out + co];
          if (bias) {
            v += bias->values[co];
          }
          const size_t out_idx =
              ((ni * c_out + co) * out_h + oh) * out_w + ow;
          output->values[out_idx] = v;
        }
      }
    }
  }

  sm_destroy(w_mat);
  sm_destroy(col);
  sm_destroy(y);

  g_last_backend = "gemm";
  return true;
}

static bool st_conv2d_mps_nchw(const FloatTensor *input,
                               const FloatTensor *weight,
                               const FloatTensor *bias,
                               const StConv2dParams *params,
                               FloatTensor *output) {
#if defined(USE_ACCELERATE_MPS) && defined(__APPLE__)
  // Placeholder for next iteration: explicit MPS conv2d kernel binding.
  // Keep stable behavior by returning false until kernel is wired.
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

static bool st_conv2d_should_use_cpu_opt(size_t n, size_t c_in, size_t c_out,
                                         size_t out_h, size_t out_w,
                                         size_t k_h, size_t k_w) {
  const size_t out_elems = n * c_out * out_h * out_w;
  const size_t kernel_volume = c_in * k_h * k_w;
  return out_elems >= 4096 && kernel_volume >= 9;
}

static bool st_conv2d_should_use_gemm(size_t n, size_t c_in, size_t c_out,
                                      size_t out_h, size_t out_w, size_t k_h,
                                      size_t k_w) {
  const double macs = (double)n * (double)c_out * (double)out_h *
                      (double)out_w * (double)c_in * (double)k_h *
                      (double)k_w;
  /* When BLAS is available, sm_gemm delegates to cblas_sgemm which is
   * highly optimised — a lower threshold pays off.  Without BLAS the
   * naive OMP fallback makes im2col + GEMM less attractive. */
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
  const double threshold = 1.0e6;
#else
  const double threshold = 2.0e8;
#endif
  return macs >= threshold;
}

static bool st_conv2d_should_use_mps(size_t n, size_t c_in, size_t c_out,
                                     size_t out_h, size_t out_w, size_t k_h,
                                     size_t k_w) {
  st_conv_init_mps_thresholds_once();

  const double macs = (double)n * (double)c_out * (double)out_h *
                      (double)out_w * (double)c_in * (double)k_h *
                      (double)k_w;
  const size_t out_elems = n * c_out * out_h * out_w;
  return macs >= g_mps_macs_threshold && out_elems >= g_mps_out_elems_threshold;
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

  /* Overflow-safe effective kernel size: dil * (k-1) + 1 */
  size_t dil_km1_h, dil_km1_w;
  if (!st_safe_mul(params->dilation_h, kernel_h - 1, &dil_km1_h) ||
      !st_safe_mul(params->dilation_w, kernel_w - 1, &dil_km1_w)) {
    return false;
  }
  const size_t eff_h = dil_km1_h + 1;
  const size_t eff_w = dil_km1_w + 1;

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

    case ST_CONV_BACKEND_GEMM:
      ok = st_conv2d_gemm_nchw(input, weight, bias, &local, output);
      if (!ok) {
        log_error("Error: st_conv2d_nchw gemm path failed.");
        return false;
      }
      return true;

    case ST_CONV_BACKEND_MPS:
      ok = st_conv2d_mps_nchw(input, weight, bias, &local, output);
      if (ok) {
        g_last_backend = "mps";
        return true;
      }
      ok = st_conv2d_gemm_nchw(input, weight, bias, &local, output);
      if (ok) {
        g_last_backend = "mps_fallback_gemm";
        return true;
      }
      ok = st_conv2d_reference_nchw(input, weight, bias, &local, output);
      if (!ok) {
        log_error("Error: st_conv2d_nchw mps fallback path failed.");
        return false;
      }
      g_last_backend = "mps_fallback_reference";
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
      /* 1x1 convolutions: always prefer direct GEMM (zero im2col cost). */
      if (st_conv2d_is_1x1_no_pad(&local, k_h, k_w)) {
        ok = st_conv2d_gemm_1x1_nchw(input, weight, bias, output);
        if (ok) {
          return true;
        }
      }

      if (st_conv2d_should_use_mps(n, c_in, c_out, out_h, out_w, k_h, k_w)) {
        ok = st_conv2d_mps_nchw(input, weight, bias, &local, output);
        if (ok) {
          g_last_backend = "mps";
          return true;
        }
      }

      ok = st_conv2d_bnns_nchw(input, weight, bias, &local, output);
      if (ok) {
        g_last_backend = "bnns";
        return true;
      }

      if (st_conv2d_should_use_gemm(n, c_in, c_out, out_h, out_w, k_h, k_w)) {
        ok = st_conv2d_gemm_nchw(input, weight, bias, &local, output);
        if (ok) {
          return true;
        }
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

bool st_conv_set_mps_thresholds(double macs_threshold,
                                size_t out_elems_threshold) {
  if (!isfinite(macs_threshold) || macs_threshold <= 0.0 ||
      out_elems_threshold == 0u) {
    return false;
  }

  g_mps_macs_threshold = macs_threshold;
  g_mps_out_elems_threshold = out_elems_threshold;
  atomic_store_explicit(&g_mps_thresholds_initialized, true,
                        memory_order_release);
  return true;
}

void st_conv_get_mps_thresholds(double *out_macs_threshold,
                                size_t *out_out_elems_threshold) {
  st_conv_init_mps_thresholds_once();
  if (out_macs_threshold != NULL) {
    *out_macs_threshold = g_mps_macs_threshold;
  }
  if (out_out_elems_threshold != NULL) {
    *out_out_elems_threshold = g_mps_out_elems_threshold;
  }
}

void st_conv_reload_mps_thresholds_from_env(void) {
  g_mps_macs_threshold = ST_CONV_MPS_MACS_THRESHOLD_DEFAULT;
  g_mps_out_elems_threshold = ST_CONV_MPS_OUT_ELEMS_THRESHOLD_DEFAULT;

  const char *macs_env = getenv("MMATRIX_ST_CONV_MPS_MACS_THRESHOLD");
  if (macs_env != NULL) {
    double parsed = 0.0;
    if (st_parse_positive_double(macs_env, &parsed)) {
      g_mps_macs_threshold = parsed;
    } else {
      log_error("Error: invalid MMATRIX_ST_CONV_MPS_MACS_THRESHOLD, using default.");
    }
  }

  const char *out_elems_env = getenv("MMATRIX_ST_CONV_MPS_OUT_ELEMS_THRESHOLD");
  if (out_elems_env != NULL) {
    size_t parsed = 0;
    if (st_parse_positive_size_t(out_elems_env, &parsed)) {
      g_mps_out_elems_threshold = parsed;
    } else {
      log_error(
          "Error: invalid MMATRIX_ST_CONV_MPS_OUT_ELEMS_THRESHOLD, using default.");
    }
  }

  atomic_store_explicit(&g_mps_thresholds_initialized, true,
                        memory_order_release);
}
