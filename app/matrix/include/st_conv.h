/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_CONV_H
#define ST_CONV_H

#include <stdbool.h>
#include <stddef.h>

#include "st.h"

typedef enum StConvBackend {
  ST_CONV_BACKEND_AUTO = 0,
  ST_CONV_BACKEND_REFERENCE = 1,
  ST_CONV_BACKEND_CPU_OPT = 2,
  ST_CONV_BACKEND_GEMM = 3,
  ST_CONV_BACKEND_MPS = 4,
  ST_CONV_BACKEND_BNNS = 5,
} StConvBackend;

typedef struct StConv2dParams {
  size_t stride_h;
  size_t stride_w;
  size_t pad_h;
  size_t pad_w;
  size_t dilation_h;
  size_t dilation_w;
  StConvBackend backend;
} StConv2dParams;

// Return default 2D convolution parameters (stride=1, pad=0, dilation=1, backend=AUTO).
StConv2dParams st_conv2d_default_params(void);

// Compute NCHW convolution output size (out_h/out_w) from input, kernel and params.
bool st_conv2d_output_hw(size_t in_h, size_t in_w, size_t kernel_h,
                         size_t kernel_w, const StConv2dParams *params,
                         size_t *out_h, size_t *out_w);

// 2D convolution for NCHW tensors: input[N,Cin,H,W], weight[Cout,Cin,Kh,Kw], optional bias[Cout].
bool st_conv2d_nchw(const FloatTensor *input, const FloatTensor *weight,
                    const FloatTensor *bias, const StConv2dParams *params,
                    FloatTensor *output);

// Return backend name used by the last st_conv2d_nchw call.
const char *st_conv2d_last_backend(void);

#endif  // ST_CONV_H
