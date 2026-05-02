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
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

static inline MPSGraphTensorData *st_mps_make_tensor_data(
    MPSGraphDevice *gDev, const float *data, void *metal_handle, size_t bytes,
    MPSShape *shape) {
  if (metal_handle) {
    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)metal_handle;
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:mtl_buf
                                                  shape:shape
                                               dataType:MPSDataTypeFloat32];
  }

  return [[MPSGraphTensorData alloc]
      initWithDevice:gDev
                data:[NSData dataWithBytesNoCopy:(void *)data
                                          length:bytes
                                    freeWhenDone:NO]
               shape:shape
            dataType:MPSDataTypeFloat32];
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

/// MPS-accelerated MaxPool2D forward (NCHW layout).
/// indices may be NULL.
/// input_metal_handle: if non-NULL, an id<MTLBuffer> (cast to void*) for
/// zero-copy GPU input. When NULL, caller-owned CPU memory is wrapped
/// without copy for duration of graph run.
bool st_maxpool2d_mps(const float *input, void *input_metal_handle,
                      size_t n, size_t c, size_t h,
                      size_t w, size_t kernel_h, size_t kernel_w,
                      size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, float *output, size_t out_h, size_t out_w);

/// MPS-accelerated AvgPool2D forward (NCHW layout).
bool st_avgpool2d_mps(const float *input, void *input_metal_handle,
                      size_t n, size_t c, size_t h,
                      size_t w, size_t kernel_h, size_t kernel_w,
                      size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, float *output, size_t out_h, size_t out_w);

/// MPS-accelerated BatchNorm2D forward (NCHW layout).
/// gamma and beta may be NULL (defaults: gamma=1, beta=0).
/// mean and var are computed and written (pre-allocated, size C).
bool st_batchnorm2d_forward_mps(const float *input, void *input_metal_handle,
                                size_t n, size_t c,
                                size_t h, size_t w, const float *gamma,
                                const float *beta, float epsilon,
                                float *output, float *mean_out,
                                float *var_out);

/* ---- Warmup: pre-populate MPSGraph cache (reduce first-run latency) ---- */

/// Pre-compiles the MPSGraph for MaxPool2D at the given shape/stride/pad.
/// Safe to call from any thread; no-op if MPS is unavailable.
void st_mps_warmup_maxpool2d(size_t n, size_t c, size_t h, size_t w,
                              size_t kh, size_t kw,
                              size_t sh, size_t sw,
                              size_t ph, size_t pw,
                              size_t oh, size_t ow);

/// Pre-compiles the MPSGraph for AvgPool2D at the given shape/stride/pad.
void st_mps_warmup_avgpool2d(size_t n, size_t c, size_t h, size_t w,
                              size_t kh, size_t kw,
                              size_t sh, size_t sw,
                              size_t ph, size_t pw,
                              size_t oh, size_t ow);

/// Pre-compiles the MPSGraph for BatchNorm2D at the given shape.
void st_mps_warmup_batchnorm2d(size_t n, size_t c, size_t h, size_t w);

#ifdef __cplusplus
}
#endif

#endif  // ST_MPS_H
