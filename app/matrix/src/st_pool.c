/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "st_pool.h"

#include <log.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- Validation helpers ---- */

static bool st_pool_is_valid_4d(const FloatTensor *t) {
  return t != NULL && t->values != NULL && t->ndim == 4 &&
         st_is_contiguous(t);
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

  if (in_h + 2 * pad_h < kernel_h || in_w + 2 * pad_w < kernel_w) {
    return false;
  }

  *out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
  *out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
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
  if (indices && !st_pool_is_valid_4d(indices)) {
    log_error("Error: st_maxpool2d_nchw indices must be valid contiguous 4D.");
    return false;
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

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *in_plane = input->values + (ni * c + ci) * h * w;

      for (size_t ohi = 0; ohi < oh; ++ohi) {
        for (size_t owi = 0; owi < ow; ++owi) {
          float max_val = -INFINITY;
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
            indices->values[out_idx] = (float)max_idx;
          }
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

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *in_plane = input->values + (ni * c + ci) * h * w;

      for (size_t ohi = 0; ohi < oh; ++ohi) {
        for (size_t owi = 0; owi < ow; ++owi) {
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
  }

  return true;
}

/* ---- Max Pool 2D Backward ---- */

bool st_maxpool2d_backward_nchw(const FloatTensor *grad_output,
                                const FloatTensor *indices, size_t input_h,
                                size_t input_w, FloatTensor *grad_input) {
  if (!st_pool_is_valid_4d(grad_output) || !st_pool_is_valid_4d(indices) ||
      !st_pool_is_valid_4d(grad_input)) {
    log_error(
        "Error: st_maxpool2d_backward_nchw expects valid contiguous 4D "
        "tensors.");
    return false;
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

  /* Zero grad_input before accumulating. */
  memset(grad_input->values, 0, grad_input->numel * sizeof(float));

  const size_t spatial_in = input_h * input_w;

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      float *gi_plane = grad_input->values + (ni * c + ci) * spatial_in;

      for (size_t ohi = 0; ohi < out_h; ++ohi) {
        for (size_t owi = 0; owi < out_w; ++owi) {
          const size_t go_idx = ((ni * c + ci) * out_h + ohi) * out_w + owi;
          const size_t max_idx = (size_t)indices->values[go_idx];

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

  /* Zero grad_input before accumulating. */
  memset(grad_input->values, 0, grad_input->numel * sizeof(float));

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      float *gi_plane = grad_input->values + (ni * c + ci) * h * w;

      for (size_t ohi = 0; ohi < out_h; ++ohi) {
        for (size_t owi = 0; owi < out_w; ++owi) {
          /* Count valid elements in this window (for correct average). */
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

          /* Distribute gradient equally to all contributing input positions. */
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

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ci = 0; ci < c; ++ci) {
      const float *plane = input->values + (ni * c + ci) * spatial;
      float sum = 0.0f;
      for (size_t i = 0; i < spatial; ++i) {
        sum += plane[i];
      }
      output->values[ni * c + ci] = sum / (float)spatial;
    }
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
