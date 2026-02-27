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

#ifdef __cplusplus
}
#endif

#endif
