/**
 * @file sv_optim.h
 * @brief Optimizer kernels for caller-owned flat float buffers.
 */

#ifndef SV_OPTIM_H
#define SV_OPTIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Apply an SGD parameter update.
 * @param params Mutable parameter buffer.
 * @param grads Gradient buffer.
 * @param n Number of elements.
 * @param learning_rate Positive learning rate.
 * @param grad_scale Scale applied to each gradient before the update.
 * @param weight_decay Decoupled weight decay; zero disables it.
 * @return true on success, false on invalid arguments.
 */
bool sv_optim_sgd(float *params, const float *grads, size_t n,
                  float learning_rate, float grad_scale, float weight_decay);

/**
 * @brief Apply a momentum update using caller-owned velocity storage.
 * @param params Mutable parameter buffer.
 * @param grads Gradient buffer.
 * @param momentum_buffer Mutable momentum buffer of length @p n.
 * @param n Number of elements.
 * @param learning_rate Positive learning rate.
 * @param momentum Momentum coefficient in `[0, 1)`.
 * @param grad_scale Scale applied to each gradient before the update.
 * @param weight_decay Decoupled weight decay; zero disables it.
 * @return true on success, false on invalid arguments.
 */
bool sv_optim_momentum(float *params, const float *grads,
                       float *momentum_buffer, size_t n, float learning_rate,
                       float momentum, float grad_scale, float weight_decay);

/**
 * @brief Apply RMSprop using caller-owned squared-gradient storage.
 * @param params Mutable parameter buffer.
 * @param grads Gradient buffer.
 * @param squared_grad_buffer Mutable second-moment buffer of length @p n.
 * @param n Number of elements.
 * @param learning_rate Positive learning rate.
 * @param beta2 RMS decay coefficient in `(0, 1)`.
 * @param epsilon Positive denominator stabilizer.
 * @param grad_scale Scale applied to each gradient before the update.
 * @param weight_decay Decoupled weight decay; zero disables it.
 * @return true on success, false on invalid arguments.
 */
bool sv_optim_rmsprop(float *params, const float *grads,
                      float *squared_grad_buffer, size_t n,
                      float learning_rate, float beta2, float epsilon,
                      float grad_scale, float weight_decay);

/**
 * @brief Apply Adam using caller-owned first- and second-moment storage.
 * @param params Mutable parameter buffer.
 * @param grads Gradient buffer.
 * @param first_moment Mutable first-moment buffer of length @p n.
 * @param second_moment Mutable second-moment buffer of length @p n.
 * @param n Number of elements.
 * @param learning_rate Positive learning rate.
 * @param beta1 First-moment decay coefficient in `(0, 1)`.
 * @param beta2 Second-moment decay coefficient in `(0, 1)`.
 * @param epsilon Positive denominator stabilizer.
 * @param step One-based optimizer step for bias correction.
 * @param grad_scale Scale applied to each gradient before the update.
 * @param weight_decay Decoupled weight decay; zero disables it.
 * @return true on success, false on invalid arguments.
 */
bool sv_optim_adam(float *params, const float *grads, float *first_moment,
                   float *second_moment, size_t n, float learning_rate,
                   float beta1, float beta2, float epsilon, uint64_t step,
                   float grad_scale, float weight_decay);

/**
 * @brief Clip gradients to an L2 norm and return the pre-clip norm.
 * @param grads Mutable gradient buffer; scaled only when norm exceeds limit.
 * @param n Number of elements.
 * @param max_norm Non-negative maximum L2 norm.
 * @return Original L2 norm, 0 for an empty buffer, or -1 for invalid arguments.
 */
float sv_clip_norm(float *grads, size_t n, float max_norm);

#endif  // SV_OPTIM_H