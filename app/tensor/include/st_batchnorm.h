/**
 * @file st_batchnorm.h
 * @brief Public API for BatchNorm2D and fused BatchNorm+ReLU in NCHW layout.
 */

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

/**
 * @brief Forward BatchNorm2D over channel dimension for NCHW tensors.
 * @param input Input tensor `[N, C, H, W]`.
 * @param gamma Optional scale tensor `[C]` (NULL means all ones).
 * @param beta Optional shift tensor `[C]` (NULL means all zeros).
 * @param epsilon Numerical stability constant.
 * @param output Output tensor `[N, C, H, W]`.
 * @param mean Optional output tensor `[C]` for channel means.
 * @param var Optional output tensor `[C]` for channel variances.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_batchnorm2d_forward(const FloatTensor *input,
                            const FloatTensor *gamma,
                            const FloatTensor *beta, float epsilon,
                            FloatTensor *output, FloatTensor *mean,
                            FloatTensor *var);

/**
 * @brief Backward BatchNorm2D for NCHW tensors.
 * @param grad_output Upstream gradient `[N, C, H, W]`.
 * @param input Original forward input `[N, C, H, W]`.
 * @param mean Channel means from forward pass `[C]`.
 * @param var Channel variances from forward pass `[C]`.
 * @param gamma Optional scale tensor `[C]`.
 * @param epsilon Same epsilon used in forward pass.
 * @param grad_input Gradient w.r.t. input `[N, C, H, W]`.
 * @param grad_gamma Optional gradient w.r.t. gamma `[C]`.
 * @param grad_beta Optional gradient w.r.t. beta `[C]`.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_batchnorm2d_backward(const FloatTensor *grad_output,
                             const FloatTensor *input,
                             const FloatTensor *mean, const FloatTensor *var,
                             const FloatTensor *gamma, float epsilon,
                             FloatTensor *grad_input, FloatTensor *grad_gamma,
                             FloatTensor *grad_beta);

/**
 * @brief Forward fused BatchNorm2D + ReLU.
 * @param input Input tensor `[N, C, H, W]`.
 * @param gamma Optional scale tensor `[C]`.
 * @param beta Optional shift tensor `[C]`.
 * @param epsilon Numerical stability constant.
 * @param output Output tensor `[N, C, H, W]` after ReLU.
 * @param mean Optional channel means `[C]`.
 * @param var Optional channel variances `[C]`.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_batchnorm2d_forward_relu(const FloatTensor *input,
                                 const FloatTensor *gamma,
                                 const FloatTensor *beta, float epsilon,
                                 FloatTensor *output, FloatTensor *mean,
                                 FloatTensor *var);

/**
 * @brief Backward fused BatchNorm2D + ReLU.
 * @param grad_output Upstream gradient.
 * @param input Original forward input.
 * @param bn_output Output of @ref st_batchnorm2d_forward_relu (post-ReLU).
 * @param mean Channel means from forward pass.
 * @param var Channel variances from forward pass.
 * @param gamma Optional scale tensor.
 * @param epsilon Same epsilon used in forward pass.
 * @param grad_input Gradient w.r.t. input.
 * @param grad_gamma Optional gradient w.r.t. gamma.
 * @param grad_beta Optional gradient w.r.t. beta.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_batchnorm2d_backward_relu(const FloatTensor *grad_output,
                                  const FloatTensor *input,
                                  const FloatTensor *bn_output,
                                  const FloatTensor *mean,
                                  const FloatTensor *var,
                                  const FloatTensor *gamma, float epsilon,
                                  FloatTensor *grad_input,
                                  FloatTensor *grad_gamma,
                                  FloatTensor *grad_beta);

#endif  // ST_BATCHNORM_H
