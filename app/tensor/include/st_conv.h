/**
 * @file st_conv.h
 * @brief Public API for 2D convolution and fused conv blocks in NCHW layout.
 */

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

/** @brief Backend selector for convolution execution. */
typedef enum StConvBackend {
  ST_CONV_BACKEND_AUTO = 0,
  ST_CONV_BACKEND_REFERENCE = 1,
  ST_CONV_BACKEND_CPU_OPT = 2,
  ST_CONV_BACKEND_GEMM = 3,
  ST_CONV_BACKEND_MPS = 4,
  ST_CONV_BACKEND_BNNS = 5,
} StConvBackend;

/** @brief Parameter bundle for 2D convolution in NCHW layout. */
typedef struct StConv2dParams {
  size_t stride_h;
  size_t stride_w;
  size_t pad_h;
  size_t pad_w;
  size_t dilation_h;
  size_t dilation_w;
  StConvBackend backend;
} StConv2dParams;

/**
 * @brief Return default convolution parameters.
 * @return Parameters with stride=1, pad=0, dilation=1, backend=AUTO.
 */
StConv2dParams st_conv2d_default_params(void);

/**
 * @brief Compute NCHW output spatial size for a 2D convolution.
 * @param in_h Input height.
 * @param in_w Input width.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param params Convolution parameters.
 * @param out_h Output height.
 * @param out_w Output width.
 * @retval true Success.
 * @retval false Invalid parameters or overflow.
 */
bool st_conv2d_output_hw(size_t in_h, size_t in_w, size_t kernel_h,
                         size_t kernel_w, const StConv2dParams *params,
                         size_t *out_h, size_t *out_w);

/**
 * @brief Execute 2D convolution for NCHW tensors.
 * @param input Input tensor `[N, Cin, H, W]`.
 * @param weight Weight tensor `[Cout, Cin, Kh, Kw]`.
 * @param bias Optional bias tensor `[Cout]` (NULL allowed).
 * @param params Convolution parameters.
 * @param output Output tensor `[N, Cout, outH, outW]`.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_conv2d_nchw(const FloatTensor *input, const FloatTensor *weight,
                    const FloatTensor *bias, const StConv2dParams *params,
                    FloatTensor *output);

/**
 * @brief Execute fused Conv2D + BatchNorm2D forward.
 * @param input Input tensor `[N, Cin, H, W]`.
 * @param weight Weight tensor `[Cout, Cin, Kh, Kw]`.
 * @param bias Optional bias tensor `[Cout]`.
 * @param params Convolution parameters.
 * @param gamma BatchNorm scale `[Cout]` (NULL means identity).
 * @param beta BatchNorm shift `[Cout]` (NULL means zero).
 * @param epsilon Numerical stability constant.
 * @param output Output tensor.
 * @param mean Optional channel means `[Cout]`.
 * @param var Optional channel variances `[Cout]`.
 * @param apply_relu Apply ReLU after BatchNorm.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_conv2d_batchnorm2d_forward_nchw(
    const FloatTensor *input, const FloatTensor *weight,
    const FloatTensor *bias, const StConv2dParams *params,
    const FloatTensor *gamma, const FloatTensor *beta, float epsilon,
    FloatTensor *output, FloatTensor *mean, FloatTensor *var,
    bool apply_relu);

/** @brief Pooling operator used in fused conv+bn+pool execution. */
typedef enum StPoolType {
  ST_POOL_MAX = 0,
  ST_POOL_AVG = 1,
} StPoolType;

/** @brief Parameter bundle for 2D pooling in NCHW layout. */
typedef struct StPool2dParams {
  StPoolType pool_type;
  size_t kernel_h;
  size_t kernel_w;
  size_t stride_h;
  size_t stride_w;
  size_t pad_h;
  size_t pad_w;
} StPool2dParams;

/**
 * @brief Execute fused Conv2D + BatchNorm2D + Pool2D forward.
 * @param input Input tensor.
 * @param weight Weight tensor.
 * @param bias Optional bias tensor.
 * @param conv_params Convolution parameters.
 * @param gamma BatchNorm scale tensor.
 * @param beta BatchNorm shift tensor.
 * @param epsilon Numerical stability constant.
 * @param pool_params Pooling parameters.
 * @param output Output tensor.
 * @param mean Optional channel means tensor.
 * @param var Optional channel variances tensor.
 * @param apply_relu Apply ReLU between BatchNorm and Pool.
 * @retval true Success.
 * @retval false Invalid input or execution failure.
 */
bool st_conv2d_batchnorm2d_pool_forward_nchw(
    const FloatTensor *input, const FloatTensor *weight,
    const FloatTensor *bias, const StConv2dParams *conv_params,
    const FloatTensor *gamma, const FloatTensor *beta, float epsilon,
    const StPool2dParams *pool_params,
    FloatTensor *output, FloatTensor *mean, FloatTensor *var,
    bool apply_relu);

/**
 * @brief Override MPS AUTO dispatch thresholds for standalone convolution.
 * @param macs_threshold Minimum MACs to use MPS in AUTO mode.
 * @param out_elems_threshold Minimum output element count to use MPS in AUTO mode.
 * @retval true Success.
 * @retval false Invalid threshold values.
 */
bool st_conv_set_mps_thresholds(double macs_threshold,
                                size_t out_elems_threshold);

/**
 * @brief Query current MPS AUTO dispatch thresholds.
 * @param out_macs_threshold Output MAC threshold (nullable).
 * @param out_out_elems_threshold Output element threshold (nullable).
 */
void st_conv_get_mps_thresholds(double *out_macs_threshold,
                                size_t *out_out_elems_threshold);

/**
 * @brief Reload MPS AUTO thresholds from environment variables.
 *
 * Variables:
 * - `MMATRIX_ST_CONV_MPS_MACS_THRESHOLD`
 * - `MMATRIX_ST_CONV_MPS_OUT_ELEMS_THRESHOLD`
 */
void st_conv_reload_mps_thresholds_from_env(void);

/**
 * @brief Return backend label used by the latest @ref st_conv2d_nchw call.
 * @return Backend name string (never NULL).
 */
const char *st_conv2d_last_backend(void);

/** @name Backward passes (training)
 *  @{ */

/**
 * @brief Compute gradient with respect to convolution input.
 */
bool st_conv2d_backward_data_nchw(const FloatTensor *grad_output,
                                  const FloatTensor *weight,
                                  const StConv2dParams *params,
                                  FloatTensor *grad_input);

/**
 * @brief Compute gradient with respect to convolution weights.
 */
bool st_conv2d_backward_weight_nchw(const FloatTensor *input,
                                    const FloatTensor *grad_output,
                                    const StConv2dParams *params,
                                    FloatTensor *grad_weight);

/**
 * @brief Compute gradient with respect to convolution bias.
 */
bool st_conv2d_backward_bias(const FloatTensor *grad_output,
                             FloatTensor *grad_bias);

/** @} */

#endif  // ST_CONV_H
