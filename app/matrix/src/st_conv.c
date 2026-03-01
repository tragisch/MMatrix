/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_conv.h"
#include "sm.h"

#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
#include <Accelerate/Accelerate.h>
#endif

#if defined(USE_ACCELERATE_MPS) && defined(__APPLE__)
#include "sm_mps.h"
#endif

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
  return mps_conv2d_nchw(
      input->values, input->shape[0],
      input->shape[1], input->shape[2], input->shape[3],
      weight->values, weight->shape[0],
      weight->shape[2], weight->shape[3],
      bias ? bias->values : NULL,
      params->stride_h, params->stride_w,
      params->pad_h, params->pad_w,
      params->dilation_h, params->dilation_w,
      output->values, output->shape[2], output->shape[3]);
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

/* ---- col2im helper (inverse of im2col) ---- */

static void st_col2im(const float *col_data, size_t c, size_t h, size_t w,
                      size_t k_h, size_t k_w, size_t stride_h, size_t stride_w,
                      size_t pad_h, size_t pad_w, size_t dil_h, size_t dil_w,
                      size_t out_h, size_t out_w, float *img_data) {
  /* col_data layout: [out_spatial, patch_size] row-major (same as forward im2col),
   * i.e. col_data[out_row * patch_size + col_idx]. */
  const size_t patch_size = c * k_h * k_w;

  for (size_t ohi = 0; ohi < out_h; ++ohi) {
    for (size_t owi = 0; owi < out_w; ++owi) {
      const size_t out_row = ohi * out_w + owi;
      for (size_t ci = 0; ci < c; ++ci) {
        for (size_t kh = 0; kh < k_h; ++kh) {
          for (size_t kw = 0; kw < k_w; ++kw) {
            const ptrdiff_t ih =
                (ptrdiff_t)(ohi * stride_h + kh * dil_h) - (ptrdiff_t)pad_h;
            const ptrdiff_t iw =
                (ptrdiff_t)(owi * stride_w + kw * dil_w) - (ptrdiff_t)pad_w;
            const size_t col_idx = ((ci * k_h + kh) * k_w + kw);

            if (ih >= 0 && iw >= 0 && (size_t)ih < h && (size_t)iw < w) {
              img_data[ci * h * w + (size_t)ih * w + (size_t)iw] +=
                  col_data[out_row * patch_size + col_idx];
            }
          }
        }
      }
    }
  }
}

/* ---- im2col helper for backward (builds col matrix for one batch element) ---- */

static void st_im2col_for_backward(const float *img_data, size_t c, size_t h,
                                   size_t w, size_t k_h, size_t k_w,
                                   size_t stride_h, size_t stride_w,
                                   size_t pad_h, size_t pad_w, size_t dil_h,
                                   size_t dil_w, size_t out_h, size_t out_w,
                                   float *col_data) {
  const size_t patch_size = c * k_h * k_w;
  const size_t out_spatial = out_h * out_w;

  for (size_t out_row = 0; out_row < out_spatial; ++out_row) {
    const size_t oh = out_row / out_w;
    const size_t ow = out_row % out_w;

    for (size_t ci = 0; ci < c; ++ci) {
      for (size_t kh = 0; kh < k_h; ++kh) {
        for (size_t kw = 0; kw < k_w; ++kw) {
          const ptrdiff_t ih =
              (ptrdiff_t)(oh * stride_h + kh * dil_h) - (ptrdiff_t)pad_h;
          const ptrdiff_t iw =
              (ptrdiff_t)(ow * stride_w + kw * dil_w) - (ptrdiff_t)pad_w;
          const size_t col_idx = (ci * k_h + kh) * k_w + kw;

          float v = 0.0f;
          if (ih >= 0 && iw >= 0 && (size_t)ih < h && (size_t)iw < w) {
            v = img_data[ci * h * w + (size_t)ih * w + (size_t)iw];
          }
          col_data[out_row * patch_size + col_idx] = v;
        }
      }
    }
  }
}

/* ---- Naive reference backward data (kept as fallback) ---- */

static bool st_conv2d_backward_data_naive(const FloatTensor *grad_output,
                                          const FloatTensor *weight,
                                          const StConv2dParams *local,
                                          FloatTensor *grad_input) {
  const size_t n = grad_output->shape[0];
  const size_t c_out = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  const size_t c_in = weight->shape[1];
  const size_t k_h = weight->shape[2];
  const size_t k_w = weight->shape[3];

  const size_t h = grad_input->shape[2];
  const size_t w = grad_input->shape[3];

  const size_t stride_h = local->stride_h;
  const size_t stride_w = local->stride_w;
  const size_t pad_h = local->pad_h;
  const size_t pad_w = local->pad_w;
  const size_t dil_h = local->dilation_h;
  const size_t dil_w = local->dilation_w;

#pragma omp parallel for schedule(static) if (n > 1)
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t co = 0; co < c_out; ++co) {
      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
          const size_t go_idx =
              ((ni * c_out + co) * out_h + oh) * out_w + ow;
          const float go_val = grad_output->values[go_idx];

          for (size_t ci = 0; ci < c_in; ++ci) {
            for (size_t kh = 0; kh < k_h; ++kh) {
              for (size_t kw = 0; kw < k_w; ++kw) {
                const ptrdiff_t ih =
                    (ptrdiff_t)(oh * stride_h + kh * dil_h) -
                    (ptrdiff_t)pad_h;
                const ptrdiff_t iw =
                    (ptrdiff_t)(ow * stride_w + kw * dil_w) -
                    (ptrdiff_t)pad_w;

                if (ih < 0 || iw < 0 || (size_t)ih >= h ||
                    (size_t)iw >= w) {
                  continue;
                }

                const size_t gi_idx =
                    ((ni * c_in + ci) * h + (size_t)ih) * w + (size_t)iw;
                const size_t w_idx =
                    ((co * c_in + ci) * k_h + kh) * k_w + kw;
                grad_input->values[gi_idx] +=
                    go_val * weight->values[w_idx];
              }
            }
          }
        }
      }
    }
  }

  return true;
}

/* ---- GEMM-based backward data: weight_T × col → col2im ---- */

static bool st_conv2d_backward_data_gemm(const FloatTensor *grad_output,
                                         const FloatTensor *weight,
                                         const StConv2dParams *local,
                                         FloatTensor *grad_input) {
  const size_t n = grad_output->shape[0];
  const size_t c_out = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  const size_t c_in = weight->shape[1];
  const size_t k_h = weight->shape[2];
  const size_t k_w = weight->shape[3];

  const size_t h = grad_input->shape[2];
  const size_t w = grad_input->shape[3];

  const size_t out_spatial = out_h * out_w;
  const size_t patch_size = c_in * k_h * k_w;

  /* Weight [Cout, Cin, Kh, Kw] viewed as row-major [c_out, patch_size].
   * In the forward pass, weight is transposed to [patch_size, c_out] and
   * col × w_fwd = y. For backward data:
   * d_col = dy × W_nchw where dy = go^T[out_spatial, c_out]
   * and W_nchw = [c_out, patch_size], giving d_col [out_spatial, patch_size]. */
  FloatMatrix w_mat = {
      .rows = c_out, .cols = patch_size, .capacity = 0,
      .values = weight->values};

  /* d_col buffer: [out_spatial, patch_size] — same layout as forward im2col */
  float *col_buf = (float *)calloc(out_spatial * patch_size, sizeof(float));
  if (!col_buf) {
    log_error("Error: st_conv2d_backward_data_gemm allocation failed.");
    return false;
  }

  FloatMatrix d_col = {
      .rows = out_spatial, .cols = patch_size,
      .capacity = out_spatial * patch_size, .values = col_buf};

  for (size_t ni = 0; ni < n; ++ni) {
    /* grad_output_n is stored NCHW as [c_out, out_spatial]. */
    FloatMatrix go_mat = {
        .rows = c_out, .cols = out_spatial, .capacity = 0,
        .values = grad_output->values + ni * c_out * out_spatial};

    /* d_col = go^T × w_nchw
     *       = [out_spatial, c_out] × [c_out, patch_size]
     *       = [out_spatial, patch_size] */
    if (!sm_gemm(&d_col, 1.0f, &go_mat, SM_TRANSPOSE, &w_mat,
                 SM_NO_TRANSPOSE, 0.0f)) {
      free(col_buf);
      log_error("Error: st_conv2d_backward_data_gemm GEMM failed.");
      return false;
    }

    /* col2im: scatter col_buf into grad_input[ni] */
    float *gi_n = grad_input->values + ni * c_in * h * w;
    memset(gi_n, 0, c_in * h * w * sizeof(float));
    st_col2im(col_buf, c_in, h, w, k_h, k_w, local->stride_h, local->stride_w,
              local->pad_h, local->pad_w, local->dilation_h, local->dilation_w,
              out_h, out_w, gi_n);
  }

  free(col_buf);
  return true;
}

/* ---- Winograd F(2×2, 3×3) backward data for stride==1, dilation==1 ---- */

/*
 * Winograd F(2,3) transforms:
 * B^T = [[1, 0, -1, 0],   G = [[1,    0,   0  ],   A^T = [[1, 1, 1, 0],
 *        [0, 1,  1, 0],        [0.5,  0.5, 0.5],          [0, 1,-1, 1]]
 *        [0,-1,  1, 0],        [0.5, -0.5, 0.5],
 *        [0, 1,  0,-1]]        [0,    0,   1  ]]
 *
 * For backward-data with 3×3 kernel, we rotate the kernel 180° (transposed
 * convolution) and apply Winograd F(2,3) on tiles.
 * Input tile: 4×4, output tile: 2×2.
 */

static bool __attribute__((unused)) st_conv2d_backward_data_winograd_3x3(
    const FloatTensor *grad_output, const FloatTensor *weight,
    const StConv2dParams *local, FloatTensor *grad_input) {
  const size_t n = grad_output->shape[0];
  const size_t c_out = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  const size_t c_in = weight->shape[1];

  const size_t h = grad_input->shape[2];
  const size_t w = grad_input->shape[3];

  /* Backward data = convolution of grad_output with rotated-180° weight
   * (full convolution). For stride==1 no-dilation, we need to pad grad_output
   * by (kernel-1) on each side, then convolve with rotated weight.
   *
   * Pad grad_output: [N, Cout, out_h+2, out_w+2] (padding by k-1 = 2 on each side)
   * Rotated weight: weight[co, ci, kh, kw] → rot_w[ci, co, 2-kh, 2-kw]
   * Then standard conv: rot_w convolved with padded_grad_output.
   *
   * For simplicity and correctness, use Winograd on this transposed conv.
   * However, implementing full Winograd tiling with all edge cases is complex.
   * We'll use a simplified approach: tile-based Winograd where possible,
   * fallback to GEMM for edge tiles.
   */

  /* Since implementing full Winograd tiling with boundary handling is
   * non-trivial and error-prone, we use the GEMM path which is already
   * highly optimized with BLAS. The Winograd path will be dispatched
   * only when the sizes are well-aligned. */

  /* Pad grad_output by 2 on each side to get full convolution */
  const size_t pad_go_h = out_h + 4;
  const size_t pad_go_w = out_w + 4;

  /* The output size of the full conv should be h × w */
  /* full conv output = pad_go - kernel + 1 = (out_h + 4) - 3 + 1 = out_h + 2 */
  /* For this to equal h, we need h = out_h + 2 which is true when
   * stride==1, pad==0: out_h = h - 3 + 1 = h-2, so h = out_h + 2. ✓
   * For pad_h != 0: out_h = h + 2*pad_h - 2, so h = out_h - 2*pad_h + 2. */

  const size_t expected_h = out_h + 2 - 2 * local->pad_h;
  const size_t expected_w = out_w + 2 - 2 * local->pad_w;

  if (h != expected_h || w != expected_w) {
    return false;  /* Dimensions don't match — fallback. */
  }

  /* Tile dimensions: input tile 4×4, output tile 2×2 */
  const size_t tile_out = 2;
  const size_t tile_in = 4;

  /* Number of tiles in each dimension (may need to handle remainder) */
  const size_t tiles_h = (h + tile_out - 1) / tile_out;
  const size_t tiles_w = (w + tile_out - 1) / tile_out;

  /* Rotate weight 180°: w_rot[ci, co, kh, kw] = weight[co, ci, 2-kh, 2-kw] */
  float *w_rot = (float *)malloc(c_in * c_out * 9 * sizeof(float));
  if (!w_rot) {
    log_error("Error: st_conv2d_backward_data_winograd_3x3 allocation failed.");
    return false;
  }

  for (size_t ci = 0; ci < c_in; ++ci) {
    for (size_t co = 0; co < c_out; ++co) {
      for (size_t kh = 0; kh < 3; ++kh) {
        for (size_t kw = 0; kw < 3; ++kw) {
          w_rot[((ci * c_out + co) * 3 + kh) * 3 + kw] =
              weight->values[((co * c_in + ci) * 3 + (2 - kh)) * 3 +
                             (2 - kw)];
        }
      }
    }
  }

  /* Transform all filters with G: Gw = G × g × G^T (for each ci,co pair)
   * G is 4×3:
   * [[1,    0,   0  ],
   *  [0.5,  0.5, 0.5],
   *  [0.5, -0.5, 0.5],
   *  [0,    0,   1  ]]
   */
  float *Gw = (float *)malloc(c_in * c_out * 16 * sizeof(float));
  if (!Gw) {
    free(w_rot);
    log_error("Error: st_conv2d_backward_data_winograd_3x3 allocation failed.");
    return false;
  }

  for (size_t ci = 0; ci < c_in; ++ci) {
    for (size_t co = 0; co < c_out; ++co) {
      const float *g = &w_rot[((ci * c_out + co) * 3) * 3];
      float tmp[4][3];

      /* tmp = G × g  (4×3 × 3×3 → 4×3) */
      for (size_t j = 0; j < 3; ++j) {
        tmp[0][j] = g[0 * 3 + j];
        tmp[1][j] = 0.5f * (g[0 * 3 + j] + g[1 * 3 + j] + g[2 * 3 + j]);
        tmp[2][j] = 0.5f * (g[0 * 3 + j] - g[1 * 3 + j] + g[2 * 3 + j]);
        tmp[3][j] = g[2 * 3 + j];
      }

      /* Gw_tile = tmp × G^T  (4×3 × 3×4 → 4×4) */
      float *gw_tile = &Gw[(ci * c_out + co) * 16];
      for (size_t i = 0; i < 4; ++i) {
        gw_tile[i * 4 + 0] = tmp[i][0];
        gw_tile[i * 4 + 1] =
            0.5f * (tmp[i][0] + tmp[i][1] + tmp[i][2]);
        gw_tile[i * 4 + 2] =
            0.5f * (tmp[i][0] - tmp[i][1] + tmp[i][2]);
        gw_tile[i * 4 + 3] = tmp[i][2];
      }
    }
  }

  free(w_rot);

  /* Pad grad_output by 2 on each side */
  float *padded_go = (float *)calloc(n * c_out * pad_go_h * pad_go_w,
                                     sizeof(float));
  if (!padded_go) {
    free(Gw);
    log_error("Error: st_conv2d_backward_data_winograd_3x3 allocation failed.");
    return false;
  }

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t co = 0; co < c_out; ++co) {
      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow_i = 0; ow_i < out_w; ++ow_i) {
          padded_go[((ni * c_out + co) * pad_go_h + (oh + 2)) * pad_go_w +
                    (ow_i + 2)] =
              grad_output->values[((ni * c_out + co) * out_h + oh) * out_w +
                                  ow_i];
        }
      }
    }
  }

  /* Process tiles */
#pragma omp parallel for schedule(static) if (n > 1)
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c_in; ++ci) {
      float *gi_plane = grad_input->values + (ni * c_in + ci) * h * w;

      for (size_t th = 0; th < tiles_h; ++th) {
        for (size_t tw = 0; tw < tiles_w; ++tw) {
          float out_tile[4] = {0};  /* 2×2 output */

          for (size_t co = 0; co < c_out; ++co) {
            /* Extract 4×4 input tile from padded grad_output */
            float d[4][4];
            const size_t base_h = th * tile_out;
            const size_t base_w = tw * tile_out;

            for (size_t i = 0; i < tile_in; ++i) {
              for (size_t j = 0; j < tile_in; ++j) {
                const size_t ph = base_h + i;
                const size_t pw = base_w + j;
                if (ph < pad_go_h && pw < pad_go_w) {
                  d[i][j] = padded_go[((ni * c_out + co) * pad_go_h + ph) *
                                          pad_go_w +
                                      pw];
                } else {
                  d[i][j] = 0.0f;
                }
              }
            }

            /* B^T × d × B  (input transform) */
            float BtdB[4][4];
            /* First: B^T × d → tmp[4][4] */
            float tmp2[4][4];
            for (size_t j = 0; j < 4; ++j) {
              tmp2[0][j] = d[0][j] - d[2][j];
              tmp2[1][j] = d[1][j] + d[2][j];
              tmp2[2][j] = -d[1][j] + d[2][j];
              tmp2[3][j] = d[1][j] - d[3][j];
            }
            /* tmp2 × B → BtdB[4][4] */
            for (size_t i = 0; i < 4; ++i) {
              BtdB[i][0] = tmp2[i][0] - tmp2[i][2];
              BtdB[i][1] = tmp2[i][1] + tmp2[i][2];
              BtdB[i][2] = -tmp2[i][1] + tmp2[i][2];
              BtdB[i][3] = tmp2[i][1] - tmp2[i][3];
            }

            /* Element-wise multiply with transformed filter */
            const float *gw_tile = &Gw[(ci * c_out + co) * 16];
            float M[4][4];
            for (size_t i = 0; i < 4; ++i) {
              for (size_t j = 0; j < 4; ++j) {
                M[i][j] = BtdB[i][j] * gw_tile[i * 4 + j];
              }
            }

            /* A^T × M × A  (output transform, 2×4 × 4×4 × 4×2 → 2×2) */
            /* A^T = [[1, 1, 1, 0],
             *        [0, 1,-1, 1]] */
            float tmp3[2][4];
            for (size_t j = 0; j < 4; ++j) {
              tmp3[0][j] = M[0][j] + M[1][j] + M[2][j];
              tmp3[1][j] = M[1][j] - M[2][j] + M[3][j];
            }
            /* tmp3 × A → 2×2 */
            out_tile[0] += tmp3[0][0] + tmp3[0][1] + tmp3[0][2];
            out_tile[1] += tmp3[0][1] - tmp3[0][2] + tmp3[0][3];
            out_tile[2] += tmp3[1][0] + tmp3[1][1] + tmp3[1][2];
            out_tile[3] += tmp3[1][1] - tmp3[1][2] + tmp3[1][3];
          }

          /* Write output tile to grad_input */
          for (size_t i = 0; i < tile_out; ++i) {
            for (size_t j = 0; j < tile_out; ++j) {
              const size_t oh_pos = th * tile_out + i;
              const size_t ow_pos = tw * tile_out + j;
              if (oh_pos < h && ow_pos < w) {
                gi_plane[oh_pos * w + ow_pos] += out_tile[i * tile_out + j];
              }
            }
          }
        }
      }
    }
  }

  free(Gw);
  free(padded_go);
  return true;
}

/* ---- Backward: gradient w.r.t. input ---- */

bool st_conv2d_backward_data_nchw(const FloatTensor *grad_output,
                                  const FloatTensor *weight,
                                  const StConv2dParams *params,
                                  FloatTensor *grad_input) {
  if (!st_conv2d_is_valid_tensor(grad_output, 4) ||
      !st_conv2d_is_valid_tensor(weight, 4) ||
      !st_conv2d_is_valid_tensor(grad_input, 4)) {
    log_error(
        "Error: st_conv2d_backward_data_nchw expects valid 4D tensors.");
    return false;
  }
  if (!st_is_contiguous(grad_output) || !st_is_contiguous(weight) ||
      !st_is_contiguous(grad_input)) {
    log_error(
        "Error: st_conv2d_backward_data_nchw requires contiguous tensors.");
    return false;
  }

  StConv2dParams local = st_conv2d_default_params();
  if (params) {
    local = *params;
  }

  const size_t n = grad_output->shape[0];
  const size_t c_out = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  const size_t w_c_out = weight->shape[0];
  const size_t c_in = weight->shape[1];
  const size_t k_h = weight->shape[2];
  const size_t k_w = weight->shape[3];

  if (c_out != w_c_out) {
    log_error("Error: st_conv2d_backward_data_nchw channel mismatch.");
    return false;
  }

  if (grad_input->shape[0] != n || grad_input->shape[1] != c_in) {
    log_error("Error: st_conv2d_backward_data_nchw grad_input shape mismatch.");
    return false;
  }

  /* Zero grad_input before accumulating. */
  memset(grad_input->values, 0, grad_input->numel * sizeof(float));

  /* Try GEMM path for larger problems. */
  const double macs = (double)n * (double)c_out * (double)out_h *
                      (double)out_w * (double)c_in * (double)k_h * (double)k_w;
  if (macs >= 1.0e3 && local.backend != ST_CONV_BACKEND_REFERENCE) {
    bool ok = st_conv2d_backward_data_gemm(grad_output, weight, &local,
                                           grad_input);
    if (ok) {
      return true;
    }
    /* Fallback to naive on GEMM failure. */
    memset(grad_input->values, 0, grad_input->numel * sizeof(float));
  }

  /* TODO: Winograd F(2,3) is currently disabled pending correctness fixes.
   * Once validated, re-enable with a high-MACs gate so GEMM is preferred
   * for medium-sized problems.
   */
#if 0
  /* Try Winograd F(2,3) path for 3×3 kernels with stride==1 and dilation==1. */
  if (k_h == 3 && k_w == 3 && local.stride_h == 1 && local.stride_w == 1 &&
      local.dilation_h == 1 && local.dilation_w == 1 &&
      local.backend != ST_CONV_BACKEND_REFERENCE) {
    bool ok = st_conv2d_backward_data_winograd_3x3(grad_output, weight, &local,
                                                   grad_input);
    if (ok) {
      return true;
    }
    /* Fallback on Winograd failure (e.g. dimension mismatch). */
    memset(grad_input->values, 0, grad_input->numel * sizeof(float));
  }
#endif

  return st_conv2d_backward_data_naive(grad_output, weight, &local,
                                       grad_input);
}

/* ---- Naive reference backward weight (kept as fallback) ---- */

static bool st_conv2d_backward_weight_naive(const FloatTensor *input,
                                            const FloatTensor *grad_output,
                                            const StConv2dParams *local,
                                            FloatTensor *grad_weight) {
  const size_t n = input->shape[0];
  const size_t c_in = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];

  const size_t c_out = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  const size_t k_h = grad_weight->shape[2];
  const size_t k_w = grad_weight->shape[3];

  const size_t stride_h = local->stride_h;
  const size_t stride_w = local->stride_w;
  const size_t pad_h = local->pad_h;
  const size_t pad_w = local->pad_w;
  const size_t dil_h = local->dilation_h;
  const size_t dil_w = local->dilation_w;

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t co = 0; co < c_out; ++co) {
      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
          const size_t go_idx =
              ((ni * c_out + co) * out_h + oh) * out_w + ow;
          const float go_val = grad_output->values[go_idx];

          for (size_t ci = 0; ci < c_in; ++ci) {
            for (size_t kh = 0; kh < k_h; ++kh) {
              for (size_t kw = 0; kw < k_w; ++kw) {
                const ptrdiff_t ih =
                    (ptrdiff_t)(oh * stride_h + kh * dil_h) -
                    (ptrdiff_t)pad_h;
                const ptrdiff_t iw =
                    (ptrdiff_t)(ow * stride_w + kw * dil_w) -
                    (ptrdiff_t)pad_w;

                if (ih < 0 || iw < 0 || (size_t)ih >= h ||
                    (size_t)iw >= w) {
                  continue;
                }

                const size_t in_idx =
                    ((ni * c_in + ci) * h + (size_t)ih) * w + (size_t)iw;
                const size_t gw_idx =
                    ((co * c_in + ci) * k_h + kh) * k_w + kw;
                grad_weight->values[gw_idx] +=
                    go_val * input->values[in_idx];
              }
            }
          }
        }
      }
    }
  }

  return true;
}

/* ---- GEMM-based backward weight: grad_output_reshaped × col_T ---- */

static bool st_conv2d_backward_weight_gemm(const FloatTensor *input,
                                           const FloatTensor *grad_output,
                                           const StConv2dParams *local,
                                           FloatTensor *grad_weight) {
  const size_t n = input->shape[0];
  const size_t c_in = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];

  const size_t c_out = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  const size_t k_h = grad_weight->shape[2];
  const size_t k_w = grad_weight->shape[3];

  const size_t out_spatial = out_h * out_w;
  const size_t patch_size = c_in * k_h * k_w;

  /* col buffer: [out_spatial, patch_size] */
  float *col_buf = (float *)malloc(out_spatial * patch_size * sizeof(float));
  if (!col_buf) {
    log_error("Error: st_conv2d_backward_weight_gemm allocation failed.");
    return false;
  }

  /* Temporary result matrix [c_out, patch_size] for accumulation per batch. */
  FloatMatrix *gw_mat = sm_create(c_out, patch_size);
  if (!gw_mat) {
    free(col_buf);
    return false;
  }

  /* Zero grad_weight accumulator. */
  memset(grad_weight->values, 0, grad_weight->numel * sizeof(float));

  for (size_t ni = 0; ni < n; ++ni) {
    /* im2col on input[ni] */
    const float *in_n = input->values + ni * c_in * h * w;
    st_im2col_for_backward(in_n, c_in, h, w, k_h, k_w, local->stride_h,
                           local->stride_w, local->pad_h, local->pad_w,
                           local->dilation_h, local->dilation_w, out_h, out_w,
                           col_buf);

    /* grad_output_n: [c_out, out_spatial] (NCHW layout — c_out planes,
     * each of size out_spatial).
     * We want: gw += grad_output_n × col^T
     *        [c_out, patch_size] = [c_out, out_spatial] × [out_spatial, patch_size]^T
     * Actually col is [out_spatial, patch_size], so col^T is [patch_size, out_spatial].
     * We need [c_out, patch_size] = go[c_out, out_spatial] × col[out_spatial, patch_size]
     * That's just go × col with no transposes! */
    FloatMatrix go_mat = {
        .rows = c_out, .cols = out_spatial, .capacity = 0,
        .values = grad_output->values + ni * c_out * out_spatial};
    FloatMatrix col_mat = {
        .rows = out_spatial, .cols = patch_size, .capacity = 0,
        .values = col_buf};

    memset(gw_mat->values, 0, c_out * patch_size * sizeof(float));
    if (!sm_gemm(gw_mat, 1.0f, &go_mat, SM_NO_TRANSPOSE, &col_mat,
                 SM_NO_TRANSPOSE, 0.0f)) {
      free(col_buf);
      sm_destroy(gw_mat);
      log_error("Error: st_conv2d_backward_weight_gemm GEMM failed.");
      return false;
    }

    /* Accumulate into grad_weight: gw_mat is [c_out, patch_size] which maps to
     * [c_out, c_in, k_h, k_w] directly (row-major). */
    for (size_t i = 0; i < c_out * patch_size; ++i) {
      grad_weight->values[i] += gw_mat->values[i];
    }
  }

  free(col_buf);
  sm_destroy(gw_mat);
  return true;
}

/* ---- Backward: gradient w.r.t. weight ---- */

bool st_conv2d_backward_weight_nchw(const FloatTensor *input,
                                    const FloatTensor *grad_output,
                                    const StConv2dParams *params,
                                    FloatTensor *grad_weight) {
  if (!st_conv2d_is_valid_tensor(input, 4) ||
      !st_conv2d_is_valid_tensor(grad_output, 4) ||
      !st_conv2d_is_valid_tensor(grad_weight, 4)) {
    log_error(
        "Error: st_conv2d_backward_weight_nchw expects valid 4D tensors.");
    return false;
  }
  if (!st_is_contiguous(input) || !st_is_contiguous(grad_output) ||
      !st_is_contiguous(grad_weight)) {
    log_error(
        "Error: st_conv2d_backward_weight_nchw requires contiguous tensors.");
    return false;
  }

  StConv2dParams local = st_conv2d_default_params();
  if (params) {
    local = *params;
  }

  const size_t n = input->shape[0];
  const size_t c_in = input->shape[1];
  (void)input->shape[2]; /* h — used only by sub-functions */
  (void)input->shape[3]; /* w — used only by sub-functions */

  const size_t c_out = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  if (grad_weight->shape[0] != c_out || grad_weight->shape[1] != c_in) {
    log_error(
        "Error: st_conv2d_backward_weight_nchw grad_weight shape mismatch.");
    return false;
  }

  const size_t k_h = grad_weight->shape[2];
  const size_t k_w = grad_weight->shape[3];

  /* Zero grad_weight before accumulating. */
  memset(grad_weight->values, 0, grad_weight->numel * sizeof(float));

  /* Try GEMM path for larger problems. */
  const double macs = (double)n * (double)c_out * (double)out_h *
                      (double)out_w * (double)c_in * (double)k_h * (double)k_w;
  if (macs >= 1.0e4 && local.backend != ST_CONV_BACKEND_REFERENCE) {
    bool ok = st_conv2d_backward_weight_gemm(input, grad_output, &local,
                                             grad_weight);
    if (ok) {
      return true;
    }
    /* Fallback to naive on GEMM failure. */
    memset(grad_weight->values, 0, grad_weight->numel * sizeof(float));
  }

  return st_conv2d_backward_weight_naive(input, grad_output, &local,
                                         grad_weight);
}

/* ---- Backward: gradient w.r.t. bias ---- */

bool st_conv2d_backward_bias(const FloatTensor *grad_output,
                             FloatTensor *grad_bias) {
  if (!st_conv2d_is_valid_tensor(grad_output, 4)) {
    log_error("Error: st_conv2d_backward_bias expects valid 4D grad_output.");
    return false;
  }
  if (grad_bias == NULL || grad_bias->values == NULL || grad_bias->ndim != 1 ||
      !st_is_contiguous(grad_bias)) {
    log_error(
        "Error: st_conv2d_backward_bias grad_bias must be contiguous 1D.");
    return false;
  }
  if (!st_is_contiguous(grad_output)) {
    log_error(
        "Error: st_conv2d_backward_bias requires contiguous grad_output.");
    return false;
  }

  const size_t n = grad_output->shape[0];
  const size_t c_out = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  if (grad_bias->shape[0] != c_out) {
    log_error("Error: st_conv2d_backward_bias grad_bias shape mismatch.");
    return false;
  }

  /* Summe über N, H, W für jeden Kanal. */
  memset(grad_bias->values, 0, c_out * sizeof(float));

  const size_t out_spatial = out_h * out_w;

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t co = 0; co < c_out; ++co) {
      const float *plane =
          grad_output->values + (ni * c_out + co) * out_spatial;
      float sum = 0.0f;
#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
      vDSP_sve(plane, 1, &sum, (vDSP_Length)out_spatial);
#else
      for (size_t i = 0; i < out_spatial; ++i) {
        sum += plane[i];
      }
#endif
      grad_bias->values[co] += sum;
    }
  }

  return true;
}

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
