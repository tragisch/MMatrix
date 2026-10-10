#ifndef SM_KERNELS_H
#define SM_KERNELS_H

#include "sm.h"

bool sm_kernel_apply_activation(float *values, size_t length,
                                SmActivation activation);
bool sm_kernel_apply_sigmoid(float *values, size_t length);
bool sm_kernel_apply_tanh(float *values, size_t length);
bool sm_kernel_apply_gelu_tanh(float *values, size_t length);
bool sm_kernel_apply_gelu_erf(float *values, size_t length);
bool sm_kernel_apply_softmax(float *values, size_t rows, size_t cols);
bool sm_kernel_add_bias_activation(float *values, const float *bias,
                                   size_t rows, size_t cols, size_t bias_rows,
                                   SmActivation activation);

bool sm_kernel_relu_backward(const float *activation, float *grad,
                             size_t length);
bool sm_kernel_sigmoid_backward(const float *activation, float *grad,
                                size_t length);
bool sm_kernel_tanh_backward(const float *activation, float *grad,
                             size_t length);
bool sm_kernel_gelu_tanh_backward(const float *input, float *grad,
                                  size_t length);
bool sm_kernel_gelu_erf_backward(const float *input, float *grad,
                                 size_t length);

#endif  // SM_KERNELS_H