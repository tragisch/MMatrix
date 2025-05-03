/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef NM_H
#define NM_H

/*
 * NeuralMatrix Library - planned neural network matrix operations.
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "sm.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

// Activation functions
float nm_relu(float x);
float nm_sigmoid(float x);
float nm_tanh(float x);
float nm_softmax_denominator(const float *vec, size_t len);

void nm_apply_relu(FloatMatrix *mat);
void nm_apply_sigmoid(FloatMatrix *mat);
void nm_apply_tanh(FloatMatrix *mat);
void nm_apply_softmax(FloatMatrix *mat);

// Loss functions
float nm_mse_loss(const FloatMatrix *predicted, const FloatMatrix *target);
float nm_cross_entropy_loss(const FloatMatrix *predicted,
                            const FloatMatrix *target);

// Forward operations
FloatMatrix *nm_linear(const FloatMatrix *input, const FloatMatrix *weights,
                       const FloatMatrix *bias);

#endif