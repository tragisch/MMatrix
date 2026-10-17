/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * Example: CNN→Dense Pipeline with Shape Operations
 *
 * Demonstrates a typical deep-learning workflow:
 * 1. Conv2D on [N, C_in, H, W]
 * 2. Global Average Pool → [N, C, 1, 1]
 * 3. Flatten to [N, C] (dense input)
 * 4. Multiple dense "layers" via matrix operations
 *
 * Uses st_shape_ops for flexible shape transformations.
 */

#include "st.h"
#include "st_conv.h"
#include "st_pool.h"
#include "st_shape_ops.h"
#include "sm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Simplified Dense Linear Layer (MatMul + Bias)
 * ============================================================================ */
typedef struct {
  FloatMatrix *weight;  /* [out_features, in_features] */
  FloatTensor *bias;    /* [out_features] */
} DenseLayer;

DenseLayer *dense_create(size_t in_features, size_t out_features) {
  DenseLayer *layer = (DenseLayer *)malloc(sizeof(DenseLayer));
  if (!layer) return NULL;

  /* Weight: [out_features, in_features] */
  size_t w_shape[2] = {out_features, in_features};
  FloatTensor *w_tensor = st_create(2, w_shape);
  if (!w_tensor) {
    free(layer);
    return NULL;
  }
  
  /* Initialize weights randomly (just fill with 0.1 for demo). */
  for (size_t i = 0; i < w_tensor->numel; ++i) {
    w_tensor->values[i] = 0.1f;
  }

  /* Expose as FloatMatrix (no-copy view). */
  layer->weight = (FloatMatrix *)malloc(sizeof(FloatMatrix));
  if (!layer->weight || !st_as_sm_view(w_tensor, layer->weight)) {
    st_destroy(w_tensor);
    free(layer);
    return NULL;
  }
  st_destroy(w_tensor);  /* Weight tensor can be freed; matrix view remains valid. */

  /* Bias: [out_features] */
  size_t b_shape[1] = {out_features};
  layer->bias = st_create(1, b_shape);
  if (!layer->bias) {
    free(layer->weight);
    free(layer);
    return NULL;
  }
  memset(layer->bias->values, 0, out_features * sizeof(float));

  return layer;
}

void dense_destroy(DenseLayer *layer) {
  if (!layer) return;
  free(layer->weight);
  st_destroy(layer->bias);
  free(layer);
}

/* Forward: output = input @ weight.T + bias */
FloatTensor *dense_forward(const DenseLayer *layer, const FloatTensor *input) {
  if (!layer || !input || input->ndim != 2) {
    return NULL;
  }

  size_t n_batch = input->shape[0];
  size_t in_feat = input->shape[1];
  size_t out_feat = layer->weight->rows;

  if (in_feat != layer->weight->cols) {
    return NULL;
  }

  /* Output: [batch, out_features] */
  size_t out_shape[2] = {n_batch, out_feat};
  FloatTensor *output = st_create(2, out_shape);
  if (!output) return NULL;

  /* Expand bias from [out_feat] to [n_batch, out_feat] for broadcasting. */
  for (size_t b = 0; b < n_batch; ++b) {
    /* Copy bias to output row. */
    memcpy(output->values + b * out_feat, layer->bias->values,
           out_feat * sizeof(float));
  }

  /* MatMul: output += input @ weight.T  (accumulate into bias) */
  for (size_t b = 0; b < n_batch; ++b) {
    for (size_t o = 0; o < out_feat; ++o) {
      float acc = 0.0f;
      for (size_t i = 0; i < in_feat; ++i) {
        float in_val = input->values[b * in_feat + i];
        float w_val = layer->weight->values[o * in_feat + i];
        acc += in_val * w_val;
      }
      output->values[b * out_feat + o] += acc;
    }
  }

  return output;
}

/* ============================================================================
 * CNN→Dense Example
 * ============================================================================ */
void example_cnn_to_dense(void) {
  printf("\n=== CNN→Dense Pipeline Example ===\n\n");

  /* Input: [N=2, C=3, H=8, W=8] (2 RGB images, 8x8) */
  size_t input_shape[4] = {2, 3, 8, 8};
  FloatTensor *input = st_create(4, input_shape);
  if (!input) return;

  /* Fill with dummy data. */
  for (size_t i = 0; i < input->numel; ++i) {
    input->values[i] = (float)(i % 10) / 10.0f;
  }
  printf("Input shape: [%zu, %zu, %zu, %zu]\n", input->shape[0], input->shape[1],
         input->shape[2], input->shape[3]);

  /* ---- Step 1: Conv2D ---- */
  StConv2dParams conv_p = st_conv2d_default_params();
  conv_p.stride_h = 2;
  conv_p.stride_w = 2;

  size_t out_h = 0, out_w = 0;
  if (!st_conv2d_output_hw(input->shape[2], input->shape[3], 3, 3, &conv_p,
                           &out_h, &out_w)) {
    printf("Conv2D shape computation failed\n");
    st_destroy(input);
    return;
  }

  /* Conv2D weights: [C_out=16, C_in=3, 3, 3] */
  size_t kernel_shape[4] = {16, 3, 3, 3};
  FloatTensor *conv_weights = st_create(4, kernel_shape);
  if (!conv_weights) {
    st_destroy(input);
    return;
  }
  for (size_t i = 0; i < conv_weights->numel; ++i) {
    conv_weights->values[i] = 0.01f;
  }

  /* Pre-allocate Conv2D output: [N=2, C_out=16, out_h, out_w] */
  size_t conv_out_shape[4] = {input->shape[0], 16, out_h, out_w};
  FloatTensor *conv_out = st_create(4, conv_out_shape);
  if (!conv_out) {
    st_destroy(conv_weights);
    st_destroy(input);
    return;
  }

  /* Conv2D forward (no bias for simplicity). */
  if (!st_conv2d_nchw(input, conv_weights, NULL, &conv_p, conv_out)) {
    printf("Conv2D forward failed\n");
    st_destroy(conv_out);
    st_destroy(conv_weights);
    st_destroy(input);
    return;
  }
  printf("After Conv2D: [%zu, %zu, %zu, %zu]\n", conv_out->shape[0],
         conv_out->shape[1], conv_out->shape[2], conv_out->shape[3]);

  /* ---- Step 2: Global Average Pool ---- */
  /* Pre-allocate pool output: [N, C, 1, 1] */
  size_t pool_out_shape[4] = {conv_out->shape[0], conv_out->shape[1], 1, 1};
  FloatTensor *pool_out = st_create(4, pool_out_shape);
  if (!pool_out) {
    st_destroy(conv_out);
    st_destroy(conv_weights);
    st_destroy(input);
    return;
  }

  if (!st_global_avgpool2d_nchw(conv_out, pool_out)) {
    printf("Global AvgPool failed\n");
    st_destroy(pool_out);
    st_destroy(conv_out);
    st_destroy(conv_weights);
    st_destroy(input);
    return;
  }
  printf("After Global AvgPool: [%zu, %zu, %zu, %zu]\n", pool_out->shape[0],
         pool_out->shape[1], pool_out->shape[2], pool_out->shape[3]);

  /* ---- Step 3: Flatten [N, C, 1, 1] → [N, C] ---- */
  FloatTensor *flattened = st_flatten(pool_out, 1, 4);  /* Flatten all but batch. */
  if (!flattened) {
    printf("Flatten failed\n");
    st_destroy(pool_out);
    st_destroy(conv_out);
    st_destroy(conv_weights);
    st_destroy(input);
    return;
  }
  printf("After Flatten: [%zu, %zu]\n", flattened->shape[0],
         flattened->shape[1]);

  /* ---- Step 4: Dense Layer 1 ---- */
  size_t hidden_size = 32;
  DenseLayer *dense1 = dense_create(flattened->shape[1], hidden_size);
  if (!dense1) {
    printf("Dense1 creation failed\n");
    st_destroy(flattened);
    st_destroy(pool_out);
    st_destroy(conv_out);
    st_destroy(conv_weights);
    st_destroy(input);
    return;
  }

  FloatTensor *dense1_out = dense_forward(dense1, flattened);
  if (!dense1_out) {
    printf("Dense1 forward failed\n");
    dense_destroy(dense1);
    st_destroy(flattened);
    st_destroy(pool_out);
    st_destroy(conv_out);
    st_destroy(conv_weights);
    st_destroy(input);
    return;
  }
  printf("After Dense1: [%zu, %zu]\n", dense1_out->shape[0],
         dense1_out->shape[1]);

  /* ---- Step 5: Dense Layer 2 (output) ---- */
  size_t num_classes = 10;
  DenseLayer *dense2 = dense_create(hidden_size, num_classes);
  if (!dense2) {
    printf("Dense2 creation failed\n");
    dense_destroy(dense1);
    st_destroy(dense1_out);
    st_destroy(flattened);
    st_destroy(pool_out);
    st_destroy(conv_out);
    st_destroy(conv_weights);
    st_destroy(input);
    return;
  }

  FloatTensor *logits = dense_forward(dense2, dense1_out);
  if (!logits) {
    printf("Dense2 forward failed\n");
    dense_destroy(dense2);
    dense_destroy(dense1);
    st_destroy(dense1_out);
    st_destroy(flattened);
    st_destroy(pool_out);
    st_destroy(conv_out);
    st_destroy(conv_weights);
    st_destroy(input);
    return;
  }
  printf("Final Output (logits): [%zu, %zu]\n", logits->shape[0],
         logits->shape[1]);

  /* ---- Cleanup ---- */
  st_destroy(logits);
  dense_destroy(dense2);
  st_destroy(dense1_out);
  dense_destroy(dense1);
  st_destroy(flattened);
  st_destroy(pool_out);
  st_destroy(conv_out);
  st_destroy(conv_weights);
  st_destroy(input);

  printf("\n✓ Pipeline completed successfully!\n\n");
}

/* ============================================================================
 * Main
 * ============================================================================ */
int main(void) {
  example_cnn_to_dense();
  return 0;
}
