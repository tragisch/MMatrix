/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef NM_H
#define NM_H

#include "sm.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>


/**************************************/
/*        Layer & Network Types       */
/**************************************/
typedef struct {
  FloatMatrix *weights;
  FloatMatrix *bias;
  void (*activation)(FloatMatrix *);
  void (*activation_derivative)(const FloatMatrix *, FloatMatrix *);
} DenseLayer;

typedef struct {
  DenseLayer *layers;
  size_t num_layers;
} NeuralNetwork;

/**************************************/
/*        Activation Functions        */
/**************************************/
void nm_apply_relu(FloatMatrix *mat);
void nm_apply_sigmoid(FloatMatrix *mat);
void nm_apply_tanh(FloatMatrix *mat);
void nm_apply_softmax(FloatMatrix *mat);
float nm_tanh(float x);
float nm_softmax_denominator(const float *vec, size_t len);

/**************************************/
/*      Activation Derivatives        */
/**************************************/
void nm_d_relu(const FloatMatrix *activation, FloatMatrix *grad_output);
void nm_d_sigmoid(const FloatMatrix *activation, FloatMatrix *grad_output);
void nm_d_tanh(const FloatMatrix *activation, FloatMatrix *grad_output);
void nm_d_softmax_crossentropy(const FloatMatrix *predicted,
                               const FloatMatrix *target,
                               FloatMatrix *grad_output);

/**************************************/
/*           Loss Functions           */
/**************************************/
float nm_mse_loss(const FloatMatrix *predicted, const FloatMatrix *target);
float nm_cross_entropy_loss(const FloatMatrix *predicted,
                            const FloatMatrix *target);

/**************************************/
/*         In-place Operations        */
/**************************************/
void nm_inplace_add_rowwise(FloatMatrix *mat, const FloatMatrix *row);

/**************************************/
/*        Matrix Transformations      */
/**************************************/
FloatMatrix *nm_sum_rows(const FloatMatrix *mat);
FloatMatrix *nm_argmax_rowwise(const FloatMatrix *mat);

/**************************************/
/*           Linear Layers            */
/**************************************/
FloatMatrix *nm_linear(const FloatMatrix *input, const FloatMatrix *weights,
                       const FloatMatrix *bias);
FloatMatrix *dense_forward(DenseLayer *layer, const FloatMatrix *input);
void dense_backward(DenseLayer *layer, const FloatMatrix *input,
                    const FloatMatrix *activation, FloatMatrix *grad_output,
                    float learning_rate);

/**************************************/
/*           Training Routines        */
/**************************************/
void train_one_epoch(NeuralNetwork *net, const FloatMatrix *X,
                     const FloatMatrix *y_true, size_t batch_size,
                     float learning_rate);

void train(NeuralNetwork *net, const FloatMatrix *X, const FloatMatrix *Y,
           size_t batch_size, float learning_rate, size_t epochs,
           bool verbose);

/**************************************/
/*             Inference              */
/**************************************/
FloatMatrix *predict(const NeuralNetwork *net, const FloatMatrix *input);

/**************************************/
/*        Evaluation Metrics          */
/**************************************/
float accuracy_score(const FloatMatrix *predicted, const FloatMatrix *target);

/**************************************/
/*           Utility Functions        */
/**************************************/

const char *nm_active_library(void);

// Binary save/load functions for neural network weights
bool save_network_bin(const char *path, const NeuralNetwork *net);
bool load_network_bin(const char *path, NeuralNetwork *net);

#endif // NM_H