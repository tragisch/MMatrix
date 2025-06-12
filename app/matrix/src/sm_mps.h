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
#else
// keine Includes für C, keine @class, keine NSString
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>

void mps_matrix_multiply(const float *mat1, size_t rows1, size_t cols1,
                         const float *mat2, size_t rows2, size_t cols2,
                         float *result);

#ifdef __cplusplus
}
#endif

#endif
