/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_MPS_H
#define ST_MPS_H

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

/// MPS-accelerated MaxPool2D forward (NCHW layout).
/// indices may be NULL.
bool st_maxpool2d_mps(const float *input, size_t n, size_t c, size_t h,
                      size_t w, size_t kernel_h, size_t kernel_w,
                      size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, float *output, size_t out_h, size_t out_w);

/// MPS-accelerated AvgPool2D forward (NCHW layout).
bool st_avgpool2d_mps(const float *input, size_t n, size_t c, size_t h,
                      size_t w, size_t kernel_h, size_t kernel_w,
                      size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, float *output, size_t out_h, size_t out_w);

/// MPS-accelerated BatchNorm2D forward (NCHW layout).
/// gamma and beta may be NULL (defaults: gamma=1, beta=0).
/// mean and var are computed and written (pre-allocated, size C).
bool st_batchnorm2d_forward_mps(const float *input, size_t n, size_t c,
                                size_t h, size_t w, const float *gamma,
                                const float *beta, float epsilon,
                                float *output, float *mean_out,
                                float *var_out);

#ifdef __cplusplus
}
#endif

#endif  // ST_MPS_H
