/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_pool.h"
#include "st_backend.h"
#include "st_buffer.h"
#include "st_dtype.h"

#include <log.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

#if defined(USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ---- Validation helpers ---- */

static bool st_pool_is_valid_4d(const FloatTensor *t) {
  return t != NULL && t->values != NULL && t->ndim == 4 &&
         st_is_contiguous(t);
}

static bool st_pool_is_valid_indices_tensor(const FloatTensor *t) {
  return st_pool_is_valid_4d(t) && t->dtype == ST_DTYPE_F32;
}

static void st_pool_indices_free(void *ptr) {
  free(ptr);
}

static bool st_pool_prepare_indices(FloatTensor *indices) {
  if (!indices) {
    return true;
  }

  if (indices->extra != NULL) {
    if (indices->extra_free != st_pool_indices_free) {
      log_error("Error: st_maxpool2d_nchw indices tensor metadata busy.");
      return false;
    }
    indices->extra_free(indices->extra);
    indices->extra = NULL;
    indices->extra_free = NULL;
  }

  if (indices->numel > SIZE_MAX / sizeof(size_t)) {
    log_error("Error: st_maxpool2d_nchw indices metadata overflow.");
    return false;
  }

  size_t *storage = (size_t *)malloc(indices->numel * sizeof(size_t));
  if (!storage) {
    log_error("Error: st_maxpool2d_nchw indices metadata allocation failed.");
    return false;
  }

  indices->extra = storage;
  indices->extra_free = st_pool_indices_free;
  return true;
}

static size_t *st_pool_indices_data(FloatTensor *indices) {
  if (indices == NULL || indices->extra == NULL ||
      indices->extra_free != st_pool_indices_free) {
    return NULL;
  }
  return (size_t *)indices->extra;
}

static const size_t *st_pool_indices_data_const(const FloatTensor *indices) {
  if (indices == NULL || indices->extra == NULL ||
      indices->extra_free != st_pool_indices_free) {
    return NULL;
  }
  return (const size_t *)indices->extra;
}

static bool st_pool_float_indices_are_exact(size_t spatial_in) {
  const size_t max_exact_index = ((size_t)1 << FLT_MANT_DIG);
  return spatial_in > 0 && spatial_in - 1 <= max_exact_index;
}

static bool st_pool_safe_mul(size_t a, size_t b, size_t *out) {
  if (out == NULL || (a != 0 && b > SIZE_MAX / a)) {
    return false;
  }
  *out = a * b;
  return true;
}

static bool st_pool_safe_add(size_t a, size_t b, size_t *out) {
  if (out == NULL || a > SIZE_MAX - b) {
    return false;
  }
  *out = a + b;
  return true;
}

/* ---- Output size computation ---- */

bool st_pool2d_output_hw(size_t in_h, size_t in_w, size_t kernel_h,
                         size_t kernel_w, size_t stride_h, size_t stride_w,
                         size_t pad_h, size_t pad_w, size_t *out_h,
                         size_t *out_w) {
  if (!out_h || !out_w || stride_h == 0 || stride_w == 0 || kernel_h == 0 ||
      kernel_w == 0) {
    return false;
  }

  size_t pad2_h = 0;
  size_t pad2_w = 0;
  size_t padded_h = 0;
  size_t padded_w = 0;
  if (!st_pool_safe_mul(pad_h, 2, &pad2_h) ||
      !st_pool_safe_mul(pad_w, 2, &pad2_w) ||
      !st_pool_safe_add(in_h, pad2_h, &padded_h) ||
      !st_pool_safe_add(in_w, pad2_w, &padded_w)) {
    return false;
  }

  if (padded_h < kernel_h || padded_w < kernel_w) {
    return false;
  }

  *out_h = (padded_h - kernel_h) / stride_h + 1;
  *out_w = (padded_w - kernel_w) / stride_w + 1;
  return true;
}

/* ---- Max Pool 2D Forward ---- */

bool st_maxpool2d_nchw(const FloatTensor *input, size_t kernel_h,
                       size_t kernel_w, size_t stride_h, size_t stride_w,
                       size_t pad_h, size_t pad_w, FloatTensor *output,
                       FloatTensor *indices) {
  if (!st_pool_is_valid_4d(input) || !st_pool_is_valid_4d(output)) {
    log_error("Error: st_maxpool2d_nchw expects valid contiguous 4D tensors.");
    return false;
  }
  if (indices && !st_pool_is_valid_indices_tensor(indices)) {
    log_error(
        "Error: st_maxpool2d_nchw indices must be valid contiguous 4D F32.");
    return false;
  }

  /* ---- bf16 promotion: convert inputs to f32, compute, convert back ---- */
  const bool need_bf16 =
      (input->dtype == ST_DTYPE_BF16) ||
      (output->dtype == ST_DTYPE_BF16);

  if (need_bf16) {
    FloatTensor *in_f32 = (input->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(input)
                              : (FloatTensor *)input;
    FloatTensor *out_f32 = st_create(output->ndim, output->shape);
    if (!in_f32 || !out_f32) {
      if (in_f32 != input) st_destroy(in_f32);
      st_destroy(out_f32);
      log_error("Error: st_maxpool2d_nchw bf16 promotion allocation failed.");
      return false;
    }

    bool ok = st_maxpool2d_nchw(in_f32, kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, out_f32, indices);

    if (ok && output->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(out_f32->values, (uint16_t *)output->values,
                          output->numel);
    } else if (ok) {
      memcpy(output->values, out_f32->values, output->numel * sizeof(float));
    }

    if (in_f32 != input) st_destroy(in_f32);
    st_destroy(out_f32);
    return ok;
  }

  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];

  size_t oh = 0, ow = 0;
  if (!st_pool2d_output_hw(h, w, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                           pad_w, &oh, &ow)) {
    log_error("Error: st_maxpool2d_nchw invalid pooling parameters.");
    return false;
  }

  if (output->shape[0] != n || output->shape[1] != c ||
      output->shape[2] != oh || output->shape[3] != ow) {
    log_error("Error: st_maxpool2d_nchw output shape mismatch.");
    return false;
  }
  if (indices &&
      (indices->shape[0] != n || indices->shape[1] != c ||
       indices->shape[2] != oh || indices->shape[3] != ow)) {
    log_error("Error: st_maxpool2d_nchw indices shape mismatch.");
    return false;
  }

  size_t *indices_data = NULL;
  if (indices) {
    if (!st_pool_prepare_indices(indices)) {
      return false;
    }
    indices_data = st_pool_indices_data(indices);
    if (!indices_data) {
      return false;
    }
  }

  /* ---- MPS dispatch via backend vtable ---- */
  {
    const StBackend *be = st_select_backend(ST_OP_MAXPOOL2D_FORWARD, input);
    if (be && be->maxpool2d_forward) {
      bool ok = be->maxpool2d_forward(input, kernel_h, kernel_w, stride_h,
                                      stride_w, pad_h, pad_w, output, indices);
      if (ok) return true;
    }
  }

  const size_t nc = n * c;

#pragma omp parallel for schedule(static) if (nc > 4)
  for (size_t nci = 0; nci < nc; ++nci) {
    const size_t ni = nci / c;
    const size_t ci = nci % c;
    const float *in_plane = input->values + (ni * c + ci) * h * w;

    for (size_t ohi = 0; ohi < oh; ++ohi) {
#ifdef __ARM_NEON
      size_t owi = 0;
      if (pad_h == 0 && pad_w == 0 && ow >= 4 && stride_w == 1 &&
          !indices) {
        for (; owi + 4 <= ow; owi += 4) {
          float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
          for (size_t kh = 0; kh < kernel_h; ++kh) {
            const size_t ih_val = ohi * stride_h + kh;
            if (ih_val >= h) break;
            for (size_t kw = 0; kw < kernel_w; ++kw) {
              const size_t base_iw = owi * stride_w + kw;
              if (base_iw + 3 < w) {
                float32x4_t vin = vld1q_f32(&in_plane[ih_val * w + base_iw]);
                vmax = vmaxq_f32(vmax, vin);
              } else {
                float tmp[4];
                for (int t = 0; t < 4; ++t) {
                  size_t iw_t = (owi + (size_t)t) * stride_w + kw;
                  tmp[t] = (iw_t < w) ? in_plane[ih_val * w + iw_t] : -FLT_MAX;
                }
                float32x4_t vin = vld1q_f32(tmp);
                vmax = vmaxq_f32(vmax, vin);
              }
            }
          }
          const size_t base_out = ((ni * c + ci) * oh + ohi) * ow + owi;
          vst1q_f32(&output->values[base_out], vmax);
        }
      }
      for (; owi < ow; ++owi) {
#else
      for (size_t owi = 0; owi < ow; ++owi) {
#endif
        float max_val = -FLT_MAX;
        size_t max_idx = 0;

        for (size_t kh = 0; kh < kernel_h; ++kh) {
          for (size_t kw = 0; kw < kernel_w; ++kw) {
            const ptrdiff_t ih =
                (ptrdiff_t)(ohi * stride_h + kh) - (ptrdiff_t)pad_h;
            const ptrdiff_t iw =
                (ptrdiff_t)(owi * stride_w + kw) - (ptrdiff_t)pad_w;

            if (ih >= 0 && iw >= 0 && (size_t)ih < h && (size_t)iw < w) {
              const size_t in_idx = (size_t)ih * w + (size_t)iw;
              const float val = in_plane[in_idx];
              if (val > max_val) {
                max_val = val;
                max_idx = in_idx;
              }
            }
          }
        }

        const size_t out_idx = ((ni * c + ci) * oh + ohi) * ow + owi;
        output->values[out_idx] = max_val;
        if (indices) {
          indices_data[out_idx] = max_idx;
          indices->values[out_idx] = (float)max_idx;
        }
      }
    }
  }

  return true;
}

/* ---- Average Pool 2D Forward ---- */

bool st_avgpool2d_nchw(const FloatTensor *input, size_t kernel_h,
                       size_t kernel_w, size_t stride_h, size_t stride_w,
                       size_t pad_h, size_t pad_w, FloatTensor *output) {
  if (!st_pool_is_valid_4d(input) || !st_pool_is_valid_4d(output)) {
    log_error("Error: st_avgpool2d_nchw expects valid contiguous 4D tensors.");
    return false;
  }

  /* ---- bf16 promotion ---- */
  const bool need_bf16 =
      (input->dtype == ST_DTYPE_BF16) ||
      (output->dtype == ST_DTYPE_BF16);

  if (need_bf16) {
    FloatTensor *in_f32 = (input->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(input)
                              : (FloatTensor *)input;
    FloatTensor *out_f32 = st_create(output->ndim, output->shape);
    if (!in_f32 || !out_f32) {
      if (in_f32 != input) st_destroy(in_f32);
      st_destroy(out_f32);
      log_error("Error: st_avgpool2d_nchw bf16 promotion allocation failed.");
      return false;
    }

    bool ok = st_avgpool2d_nchw(in_f32, kernel_h, kernel_w, stride_h, stride_w,
                                pad_h, pad_w, out_f32);

    if (ok && output->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(out_f32->values, (uint16_t *)output->values,
                          output->numel);
    } else if (ok) {
      memcpy(output->values, out_f32->values, output->numel * sizeof(float));
    }

    if (in_f32 != input) st_destroy(in_f32);
    st_destroy(out_f32);
    return ok;
  }

  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];

  size_t oh = 0, ow = 0;
  if (!st_pool2d_output_hw(h, w, kernel_h, kernel_w, stride_h, stride_w, pad_h,
                           pad_w, &oh, &ow)) {
    log_error("Error: st_avgpool2d_nchw invalid pooling parameters.");
    return false;
  }

  if (output->shape[0] != n || output->shape[1] != c ||
      output->shape[2] != oh || output->shape[3] != ow) {
    log_error("Error: st_avgpool2d_nchw output shape mismatch.");
    return false;
  }

  /* ---- MPS dispatch via backend vtable ---- */
  {
    const StBackend *be = st_select_backend(ST_OP_AVGPOOL2D_FORWARD, input);
    if (be && be->avgpool2d_forward) {
      bool ok = be->avgpool2d_forward(input, kernel_h, kernel_w, stride_h,
                                      stride_w, pad_h, pad_w, output);
      if (ok) return true;
    }
  }

  const size_t nc_avg = n * c;

#pragma omp parallel for schedule(static) if (nc_avg > 4)
  for (size_t nci = 0; nci < nc_avg; ++nci) {
    const size_t ni = nci / c;
    const size_t ci = nci % c;
    const float *in_plane = input->values + (ni * c + ci) * h * w;

    for (size_t ohi = 0; ohi < oh; ++ohi) {
#ifdef __ARM_NEON
      size_t owi_avg = 0;
      if (pad_h == 0 && pad_w == 0 && ow >= 4 && stride_w == 1) {
        const float inv_count = 1.0f / (float)(kernel_h * kernel_w);
        const float32x4_t vinv = vdupq_n_f32(inv_count);
        for (; owi_avg + 4 <= ow; owi_avg += 4) {
          float32x4_t vsum = vdupq_n_f32(0.0f);
          bool all_inside = true;
          for (size_t kh = 0; kh < kernel_h && all_inside; ++kh) {
            const size_t ih_val = ohi * stride_h + kh;
            if (ih_val >= h) { all_inside = false; break; }
            for (size_t kw = 0; kw < kernel_w; ++kw) {
              const size_t base_iw = owi_avg * stride_w + kw;
              if (base_iw + 3 < w) {
                float32x4_t vin = vld1q_f32(&in_plane[ih_val * w + base_iw]);
                vsum = vaddq_f32(vsum, vin);
              } else {
                all_inside = false;
                break;
              }
            }
          }
          if (all_inside) {
            vsum = vmulq_f32(vsum, vinv);
            const size_t base_out = ((ni * c + ci) * oh + ohi) * ow + owi_avg;
            vst1q_f32(&output->values[base_out], vsum);
          } else {
            owi_avg -= 0;
            break;
          }
        }
      }
      for (size_t owi = owi_avg; owi < ow; ++owi) {
#else
      for (size_t owi = 0; owi < ow; ++owi) {
#endif
        float sum = 0.0f;
        size_t count = 0;

        for (size_t kh = 0; kh < kernel_h; ++kh) {
          for (size_t kw = 0; kw < kernel_w; ++kw) {
            const ptrdiff_t ih =
                (ptrdiff_t)(ohi * stride_h + kh) - (ptrdiff_t)pad_h;
            const ptrdiff_t iw =
                (ptrdiff_t)(owi * stride_w + kw) - (ptrdiff_t)pad_w;

            if (ih >= 0 && iw >= 0 && (size_t)ih < h && (size_t)iw < w) {
              sum += in_plane[(size_t)ih * w + (size_t)iw];
              ++count;
            }
          }
        }

        const size_t out_idx = ((ni * c + ci) * oh + ohi) * ow + owi;
        output->values[out_idx] = (count > 0) ? sum / (float)count : 0.0f;
      }
    }
  }

  return true;
}

/* ---- Max Pool 2D Backward ---- */

bool st_maxpool2d_backward_nchw(const FloatTensor *grad_output,
                                const FloatTensor *indices, size_t input_h,
                                size_t input_w, FloatTensor *grad_input) {
  if (!st_pool_is_valid_4d(grad_output) ||
      !st_pool_is_valid_indices_tensor(indices) ||
      !st_pool_is_valid_4d(grad_input)) {
    log_error(
        "Error: st_maxpool2d_backward_nchw expects valid contiguous 4D "
        "tensors and F32 indices.");
    return false;
  }

  /* ---- bf16 promotion ---- */
  const bool need_bf16 =
      (grad_output->dtype == ST_DTYPE_BF16) ||
      (grad_input->dtype == ST_DTYPE_BF16);

  if (need_bf16) {
    FloatTensor *go_f32 = (grad_output->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(grad_output)
                              : (FloatTensor *)grad_output;
    FloatTensor *gi_f32 = st_create(grad_input->ndim, grad_input->shape);
    if (!go_f32 || !gi_f32) {
      if (go_f32 != grad_output) st_destroy(go_f32);
      st_destroy(gi_f32);
      log_error(
          "Error: st_maxpool2d_backward_nchw bf16 promotion allocation "
          "failed.");
      return false;
    }

    bool ok = st_maxpool2d_backward_nchw(go_f32, indices, input_h, input_w,
                                         gi_f32);

    if (ok && grad_input->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(gi_f32->values, (uint16_t *)grad_input->values,
                          grad_input->numel);
    } else if (ok) {
      memcpy(grad_input->values, gi_f32->values,
             grad_input->numel * sizeof(float));
    }

    if (go_f32 != grad_output) st_destroy(go_f32);
    st_destroy(gi_f32);
    return ok;
  }

  const size_t n = grad_output->shape[0];
  const size_t c = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  if (grad_input->shape[0] != n || grad_input->shape[1] != c ||
      grad_input->shape[2] != input_h || grad_input->shape[3] != input_w) {
    log_error("Error: st_maxpool2d_backward_nchw grad_input shape mismatch.");
    return false;
  }
  if (indices->shape[0] != n || indices->shape[1] != c ||
      indices->shape[2] != out_h || indices->shape[3] != out_w) {
    log_error("Error: st_maxpool2d_backward_nchw indices shape mismatch.");
    return false;
  }

  memset(grad_input->values, 0, grad_input->numel * sizeof(float));

  const size_t spatial_in = input_h * input_w;
  const size_t *indices_data = st_pool_indices_data_const(indices);
  if (!indices_data && !st_pool_float_indices_are_exact(spatial_in)) {
    log_error("Error: st_maxpool2d_backward_nchw needs precise indices.");
    return false;
  }

#pragma omp parallel for collapse(2) schedule(static) if (n * c > 4)
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      float *gi_plane = grad_input->values + (ni * c + ci) * spatial_in;

      for (size_t ohi = 0; ohi < out_h; ++ohi) {
        for (size_t owi = 0; owi < out_w; ++owi) {
          const size_t go_idx = ((ni * c + ci) * out_h + ohi) * out_w + owi;
          const size_t max_idx = indices_data ? indices_data[go_idx]
                                              : (size_t)indices->values[go_idx];

          if (max_idx < spatial_in) {
            gi_plane[max_idx] += grad_output->values[go_idx];
          }
        }
      }
    }
  }

  return true;
}

/* ---- Average Pool 2D Backward ---- */

bool st_avgpool2d_backward_nchw(const FloatTensor *grad_output,
                                size_t kernel_h, size_t kernel_w,
                                size_t stride_h, size_t stride_w, size_t pad_h,
                                size_t pad_w, FloatTensor *grad_input) {
  if (!st_pool_is_valid_4d(grad_output) || !st_pool_is_valid_4d(grad_input)) {
    log_error(
        "Error: st_avgpool2d_backward_nchw expects valid contiguous 4D "
        "tensors.");
    return false;
  }

  /* ---- bf16 promotion ---- */
  const bool need_bf16_promotion =
      (grad_output->dtype == ST_DTYPE_BF16) ||
      (grad_input->dtype == ST_DTYPE_BF16);

  if (need_bf16_promotion) {
    FloatTensor *go_f32 = (grad_output->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(grad_output)
                              : (FloatTensor *)grad_output;
    FloatTensor *gi_f32 = st_create(grad_input->ndim, grad_input->shape);
    if (!go_f32 || !gi_f32) {
      if (go_f32 != grad_output) st_destroy(go_f32);
      st_destroy(gi_f32);
      log_error(
          "Error: st_avgpool2d_backward_nchw bf16 promotion allocation "
          "failed.");
      return false;
    }

    bool ok = st_avgpool2d_backward_nchw(go_f32, kernel_h, kernel_w, stride_h,
                                         stride_w, pad_h, pad_w, gi_f32);

    if (ok && grad_input->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(gi_f32->values, (uint16_t *)grad_input->values,
                          grad_input->numel);
    } else if (ok) {
      memcpy(grad_input->values, gi_f32->values,
             grad_input->numel * sizeof(float));
    }

    if (go_f32 != grad_output) st_destroy(go_f32);
    st_destroy(gi_f32);
    return ok;
  }

  const size_t n = grad_output->shape[0];
  const size_t c = grad_output->shape[1];
  const size_t out_h = grad_output->shape[2];
  const size_t out_w = grad_output->shape[3];

  const size_t h = grad_input->shape[2];
  const size_t w = grad_input->shape[3];

  if (grad_input->shape[0] != n || grad_input->shape[1] != c) {
    log_error("Error: st_avgpool2d_backward_nchw batch/channel mismatch.");
    return false;
  }

  memset(grad_input->values, 0, grad_input->numel * sizeof(float));

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      float *gi_plane = grad_input->values + (ni * c + ci) * h * w;

      for (size_t ohi = 0; ohi < out_h; ++ohi) {
        for (size_t owi = 0; owi < out_w; ++owi) {
          size_t count = 0;
          for (size_t kh = 0; kh < kernel_h; ++kh) {
            for (size_t kw = 0; kw < kernel_w; ++kw) {
              const ptrdiff_t ih =
                  (ptrdiff_t)(ohi * stride_h + kh) - (ptrdiff_t)pad_h;
              const ptrdiff_t iw =
                  (ptrdiff_t)(owi * stride_w + kw) - (ptrdiff_t)pad_w;
              if (ih >= 0 && iw >= 0 && (size_t)ih < h && (size_t)iw < w) {
                ++count;
              }
            }
          }

          if (count == 0) {
            continue;
          }

          const size_t go_idx = ((ni * c + ci) * out_h + ohi) * out_w + owi;
          const float grad_val =
              grad_output->values[go_idx] / (float)count;

          for (size_t kh = 0; kh < kernel_h; ++kh) {
            for (size_t kw = 0; kw < kernel_w; ++kw) {
              const ptrdiff_t ih =
                  (ptrdiff_t)(ohi * stride_h + kh) - (ptrdiff_t)pad_h;
              const ptrdiff_t iw =
                  (ptrdiff_t)(owi * stride_w + kw) - (ptrdiff_t)pad_w;
              if (ih >= 0 && iw >= 0 && (size_t)ih < h && (size_t)iw < w) {
                gi_plane[(size_t)ih * w + (size_t)iw] += grad_val;
              }
            }
          }
        }
      }
    }
  }

  return true;
}

/* ---- Global Average Pool 2D Forward ---- */

bool st_global_avgpool2d_nchw(const FloatTensor *input, FloatTensor *output) {
  if (!st_pool_is_valid_4d(input) || !st_pool_is_valid_4d(output)) {
    log_error(
        "Error: st_global_avgpool2d_nchw expects valid contiguous 4D tensors.");
    return false;
  }

  /* ---- bf16 promotion ---- */
  const bool need_bf16_promotion =
      (input->dtype == ST_DTYPE_BF16) ||
      (output->dtype == ST_DTYPE_BF16);

  if (need_bf16_promotion) {
    FloatTensor *in_f32 = (input->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(input)
                              : (FloatTensor *)input;
    FloatTensor *out_f32 = st_create(output->ndim, output->shape);
    if (!in_f32 || !out_f32) {
      if (in_f32 != input) st_destroy(in_f32);
      st_destroy(out_f32);
      log_error(
          "Error: st_global_avgpool2d_nchw bf16 promotion allocation "
          "failed.");
      return false;
    }

    bool ok = st_global_avgpool2d_nchw(in_f32, out_f32);

    if (ok && output->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(out_f32->values, (uint16_t *)output->values,
                          output->numel);
    } else if (ok) {
      memcpy(output->values, out_f32->values, output->numel * sizeof(float));
    }

    if (in_f32 != input) st_destroy(in_f32);
    st_destroy(out_f32);
    return ok;
  }

  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];
  const size_t spatial = h * w;

  if (output->shape[0] != n || output->shape[1] != c ||
      output->shape[2] != 1 || output->shape[3] != 1) {
    log_error("Error: st_global_avgpool2d_nchw output must be [N,C,1,1].");
    return false;
  }

  const size_t nc_gfwd = n * c;

#pragma omp parallel for schedule(static) if (nc_gfwd > 8)
  for (size_t nci = 0; nci < nc_gfwd; ++nci) {
    const size_t ni = nci / c;
    const size_t ci = nci % c;
    const float *plane = input->values + (ni * c + ci) * spatial;
    float sum = 0.0f;
#if defined(USE_ACCELERATE)
    vDSP_sve(plane, 1, &sum, (vDSP_Length)spatial);
#else
    for (size_t i = 0; i < spatial; ++i) {
      sum += plane[i];
    }
#endif
    output->values[ni * c + ci] = sum / (float)spatial;
  }

  return true;
}

/* ---- Global Average Pool 2D Backward ---- */

bool st_global_avgpool2d_backward_nchw(const FloatTensor *grad_output,
                                       FloatTensor *grad_input) {
  if (!st_pool_is_valid_4d(grad_output) || !st_pool_is_valid_4d(grad_input)) {
    log_error(
        "Error: st_global_avgpool2d_backward_nchw expects valid contiguous 4D "
        "tensors.");
    return false;
  }

  /* ---- bf16 promotion ---- */
  const bool need_bf16_promotion =
      (grad_output->dtype == ST_DTYPE_BF16) ||
      (grad_input->dtype == ST_DTYPE_BF16);

  if (need_bf16_promotion) {
    FloatTensor *go_f32 = (grad_output->dtype == ST_DTYPE_BF16)
                              ? st_to_f32(grad_output)
                              : (FloatTensor *)grad_output;
    FloatTensor *gi_f32 = st_create(grad_input->ndim, grad_input->shape);
    if (!go_f32 || !gi_f32) {
      if (go_f32 != grad_output) st_destroy(go_f32);
      st_destroy(gi_f32);
      log_error(
          "Error: st_global_avgpool2d_backward_nchw bf16 promotion allocation "
          "failed.");
      return false;
    }

    bool ok = st_global_avgpool2d_backward_nchw(go_f32, gi_f32);

    if (ok && grad_input->dtype == ST_DTYPE_BF16) {
      st_f32_to_bf16_bulk(gi_f32->values, (uint16_t *)grad_input->values,
                          grad_input->numel);
    } else if (ok) {
      memcpy(grad_input->values, gi_f32->values,
             grad_input->numel * sizeof(float));
    }

    if (go_f32 != grad_output) st_destroy(go_f32);
    st_destroy(gi_f32);
    return ok;
  }

  const size_t n = grad_input->shape[0];
  const size_t c = grad_input->shape[1];
  const size_t h = grad_input->shape[2];
  const size_t w = grad_input->shape[3];
  const size_t spatial = h * w;

  if (grad_output->shape[0] != n || grad_output->shape[1] != c ||
      grad_output->shape[2] != 1 || grad_output->shape[3] != 1) {
    log_error(
        "Error: st_global_avgpool2d_backward_nchw grad_output must be "
        "[N,C,1,1].");
    return false;
  }

  const float inv_spatial = 1.0f / (float)spatial;

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float grad_val = grad_output->values[ni * c + ci] * inv_spatial;
      float *plane = grad_input->values + (ni * c + ci) * spatial;
      for (size_t i = 0; i < spatial; ++i) {
        plane[i] = grad_val;
      }
    }
  }

  return true;
}
