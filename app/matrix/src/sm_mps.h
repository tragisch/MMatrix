/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef SM_MPS_H
#define SM_MPS_H

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#else
// keine Includes für C, keine @class, keine NSString
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>
#include <stdbool.h>

/// Returns the shared MTLDevice (as opaque pointer). Thread-safe.
void *mps_get_shared_device(void);

/// Returns the shared MTLCommandQueue (as opaque pointer). Thread-safe.
void *mps_get_shared_command_queue(void);

bool mps_matrix_multiply(const float *mat1, size_t rows1, size_t cols1,
                         const float *mat2, size_t rows2, size_t cols2,
                         float *result);

bool mps_matrix_multiply_ex(const float *mat1, size_t rows1, size_t cols1,
                            bool transpose_left, const float *mat2,
                            size_t rows2, size_t cols2, bool transpose_right,
                            float alpha, float beta, float *result,
                            size_t result_rows, size_t result_cols);

/// GPU-accelerated 2D convolution via MPSGraph.
/// Input layout NCHW, weight layout OIHW (C_out, C_in, K_h, K_w).
/// Bias may be NULL.  Returns false on parameter or GPU errors.
bool mps_conv2d_nchw(const float *input, size_t n,
                     size_t c_in, size_t h_in, size_t w_in,
                     const float *weight, size_t c_out,
                     size_t k_h, size_t k_w, const float *bias,
                     size_t stride_h, size_t stride_w,
                     size_t pad_h, size_t pad_w,
                     size_t dil_h, size_t dil_w,
                     float *output, size_t h_out, size_t w_out);

#ifdef __cplusplus
}
#endif

#endif
