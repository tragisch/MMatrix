/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_BATCHNORM_H
#define ST_BATCHNORM_H

#include <stdbool.h>
#include <stddef.h>

#include "st.h"

// ---- Batch Normalization 2D (NCHW) ----

// Forward pass for Batch Normalization over channel dimension.
// input   [N, C, H, W]
// gamma   [C] — learnable scale (may be NULL for gamma=1)
// beta    [C] — learnable shift (may be NULL for beta=0)
// epsilon  small constant for numerical stability (e.g. 1e-5)
// output  [N, C, H, W] — normalized, scaled, shifted
// mean    [C] — computed channel means   (must be pre-allocated, written)
// var     [C] — computed channel variances (must be pre-allocated, written)
bool st_batchnorm2d_forward(const FloatTensor *input,
                            const FloatTensor *gamma,
                            const FloatTensor *beta, float epsilon,
                            FloatTensor *output, FloatTensor *mean,
                            FloatTensor *var);

// Backward pass for Batch Normalization.
// grad_output [N, C, H, W]
// input       [N, C, H, W] — original input from forward
// mean        [C] — from forward
// var         [C] — from forward
// gamma       [C] — learnable scale (may be NULL for gamma=1)
// epsilon      same epsilon used in forward
// grad_input  [N, C, H, W] — gradient w.r.t. input (pre-allocated)
// grad_gamma  [C]           — gradient w.r.t. gamma (pre-allocated; may be NULL)
// grad_beta   [C]           — gradient w.r.t. beta  (pre-allocated; may be NULL)
bool st_batchnorm2d_backward(const FloatTensor *grad_output,
                             const FloatTensor *input,
                             const FloatTensor *mean, const FloatTensor *var,
                             const FloatTensor *gamma, float epsilon,
                             FloatTensor *grad_input, FloatTensor *grad_gamma,
                             FloatTensor *grad_beta);

#endif  // ST_BATCHNORM_H
