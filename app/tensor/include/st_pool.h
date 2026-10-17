/**
 * @file st_pool.h
 * @brief Public API for pooling operators in NCHW layout.
 */

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

/**
 * @brief Compute 2D pooling output size.
 * @param in_h Input height.
 * @param in_w Input width.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride_h Vertical stride.
 * @param stride_w Horizontal stride.
 * @param pad_h Vertical symmetric padding.
 * @param pad_w Horizontal symmetric padding.
 * @param out_h Output height.
 * @param out_w Output width.
 * @retval true Success.
 * @retval false Invalid parameters or overflow.
 */
bool st_pool2d_output_hw(size_t in_h, size_t in_w, size_t kernel_h,
                         size_t kernel_w, size_t stride_h, size_t stride_w,
                         size_t pad_h, size_t pad_w, size_t *out_h,
                         size_t *out_w);

/**
 * @brief Forward max pooling for NCHW tensors.
 * @param input Input tensor `[N, C, H, W]`.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride_h Vertical stride.
 * @param stride_w Horizontal stride.
 * @param pad_h Vertical symmetric padding.
 * @param pad_w Horizontal symmetric padding.
 * @param output Output tensor `[N, C, outH, outW]`.
 * @param indices Optional tensor `[N, C, outH, outW]` storing max indices.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_maxpool2d_nchw(const FloatTensor *input, size_t kernel_h,
                       size_t kernel_w, size_t stride_h, size_t stride_w,
                       size_t pad_h, size_t pad_w, FloatTensor *output,
                       FloatTensor *indices);

/**
 * @brief Forward average pooling for NCHW tensors.
 * @param input Input tensor `[N, C, H, W]`.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride_h Vertical stride.
 * @param stride_w Horizontal stride.
 * @param pad_h Vertical symmetric padding.
 * @param pad_w Horizontal symmetric padding.
 * @param output Output tensor `[N, C, outH, outW]`.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_avgpool2d_nchw(const FloatTensor *input, size_t kernel_h,
                       size_t kernel_w, size_t stride_h, size_t stride_w,
                       size_t pad_h, size_t pad_w, FloatTensor *output);

/**
 * @brief Backward max pooling for NCHW tensors.
 * @param grad_output Upstream gradient `[N, C, outH, outW]`.
 * @param indices Saved max indices from forward pass.
 * @param input_h Original input height.
 * @param input_w Original input width.
 * @param grad_input Output gradient buffer `[N, C, H, W]` (written/accumulated).
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_maxpool2d_backward_nchw(const FloatTensor *grad_output,
                                const FloatTensor *indices, size_t input_h,
                                size_t input_w, FloatTensor *grad_input);

/**
 * @brief Backward average pooling for NCHW tensors.
 * @param grad_output Upstream gradient `[N, C, outH, outW]`.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride_h Vertical stride.
 * @param stride_w Horizontal stride.
 * @param pad_h Vertical symmetric padding.
 * @param pad_w Horizontal symmetric padding.
 * @param grad_input Output gradient buffer `[N, C, H, W]` (written/accumulated).
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_avgpool2d_backward_nchw(const FloatTensor *grad_output,
                                size_t kernel_h, size_t kernel_w,
                                size_t stride_h, size_t stride_w, size_t pad_h,
                                size_t pad_w, FloatTensor *grad_input);

/**
 * @brief Global average pooling over spatial dimensions.
 * @param input Input tensor `[N, C, H, W]`.
 * @param output Output tensor `[N, C, 1, 1]`.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_global_avgpool2d_nchw(const FloatTensor *input, FloatTensor *output);

/**
 * @brief Backward global average pooling.
 * @param grad_output Upstream gradient `[N, C, 1, 1]`.
 * @param grad_input Output gradient `[N, C, H, W]`.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_global_avgpool2d_backward_nchw(const FloatTensor *grad_output,
                                       FloatTensor *grad_input);

#endif  // ST_POOL_H
