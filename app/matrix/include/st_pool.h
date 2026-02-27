/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_POOL_H
#define ST_POOL_H

#include <stdbool.h>
#include <stddef.h>

#include "st.h"

// ---- Pooling output size helper ----

// Compute 2D pooling output size (out_h, out_w) from input, kernel and params.
bool st_pool2d_output_hw(size_t in_h, size_t in_w, size_t kernel_h,
                         size_t kernel_w, size_t stride_h, size_t stride_w,
                         size_t pad_h, size_t pad_w, size_t *out_h,
                         size_t *out_w);

// ---- Forward ----

// Max-pool 2D for NCHW tensors.
// input  [N, C, H, W]
// output [N, C, out_H, out_W]
// indices [N, C, out_H, out_W]  — linear index of max within each window
//         (stored as float for API uniformity; cast to size_t for use).
//         Pass NULL if indices are not needed (e.g. inference-only).
bool st_maxpool2d_nchw(const FloatTensor *input, size_t kernel_h,
                       size_t kernel_w, size_t stride_h, size_t stride_w,
                       size_t pad_h, size_t pad_w, FloatTensor *output,
                       FloatTensor *indices);

// Average-pool 2D for NCHW tensors.
// input  [N, C, H, W]
// output [N, C, out_H, out_W]
bool st_avgpool2d_nchw(const FloatTensor *input, size_t kernel_h,
                       size_t kernel_w, size_t stride_h, size_t stride_w,
                       size_t pad_h, size_t pad_w, FloatTensor *output);

// ---- Backward ----

// Backward pass for max-pool 2D.
// grad_output [N, C, out_H, out_W]
// indices     [N, C, out_H, out_W]  — from st_maxpool2d_nchw forward
// grad_input  [N, C, H, W]         — accumulated gradients (zeroed by caller)
bool st_maxpool2d_backward_nchw(const FloatTensor *grad_output,
                                const FloatTensor *indices, size_t input_h,
                                size_t input_w, FloatTensor *grad_input);

// Backward pass for average-pool 2D.
// grad_output [N, C, out_H, out_W]
// grad_input  [N, C, H, W]         — accumulated gradients (zeroed by caller)
bool st_avgpool2d_backward_nchw(const FloatTensor *grad_output,
                                size_t kernel_h, size_t kernel_w,
                                size_t stride_h, size_t stride_w, size_t pad_h,
                                size_t pad_w, FloatTensor *grad_input);

// Global average-pool over spatial dims: [N,C,H,W] → [N,C,1,1].
bool st_global_avgpool2d_nchw(const FloatTensor *input, FloatTensor *output);

// Backward for global average-pool: [N,C,1,1] → [N,C,H,W].
bool st_global_avgpool2d_backward_nchw(const FloatTensor *grad_output,
                                       FloatTensor *grad_input);

#endif  // ST_POOL_H
