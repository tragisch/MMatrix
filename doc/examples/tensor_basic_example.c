#include <stddef.h>
#include <stdio.h>

#include "st.h"
#include "st_batchnorm.h"
#include "st_conv.h"
#include "st_pool.h"

static int init_input(FloatTensor *input) {
  for (size_t h = 0; h < 4; ++h) {
    for (size_t w = 0; w < 4; ++w) {
      const size_t idx[4] = {0, 0, h, w};
      const float value = (float)(h * 4 + w + 1) / 16.0f;
      if (!st_set(input, idx, value)) {
        return 0;
      }
    }
  }
  return 1;
}

static int init_weights(FloatTensor *weight) {
  for (size_t oc = 0; oc < 2; ++oc) {
    for (size_t ic = 0; ic < 1; ++ic) {
      for (size_t kh = 0; kh < 3; ++kh) {
        for (size_t kw = 0; kw < 3; ++kw) {
          const size_t idx[4] = {oc, ic, kh, kw};
          const float value = 0.03f * (float)(oc + 1) +
                              0.005f * (float)(kh * 3 + kw + 1);
          if (!st_set(weight, idx, value)) {
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

int main(void) {
  const size_t input_shape[4] = {1, 1, 4, 4};
  const size_t weight_shape[4] = {2, 1, 3, 3};
  const size_t channel_shape[1] = {2};

  FloatTensor *input = st_create(4, input_shape);
  FloatTensor *weight = st_create(4, weight_shape);
  FloatTensor *gamma = st_create(1, channel_shape);
  FloatTensor *beta = st_create(1, channel_shape);

  if (input == NULL || weight == NULL || gamma == NULL || beta == NULL) {
    fprintf(stderr, "Failed to allocate tensors.\n");
    st_destroy(beta);
    st_destroy(gamma);
    st_destroy(weight);
    st_destroy(input);
    return 1;
  }

  if (!init_input(input) || !init_weights(weight) || !st_fill(gamma, 1.0f) ||
      !st_fill(beta, 0.0f)) {
    fprintf(stderr, "Failed to initialize tensors.\n");
    st_destroy(beta);
    st_destroy(gamma);
    st_destroy(weight);
    st_destroy(input);
    return 2;
  }

  StConv2dParams conv_params = st_conv2d_default_params();
  conv_params.pad_h = 1;
  conv_params.pad_w = 1;

  size_t out_h = 0;
  size_t out_w = 0;
  if (!st_conv2d_output_hw(4, 4, 3, 3, &conv_params, &out_h, &out_w)) {
    fprintf(stderr, "Failed to compute conv output shape.\n");
    st_destroy(beta);
    st_destroy(gamma);
    st_destroy(weight);
    st_destroy(input);
    return 3;
  }

  const size_t conv_shape[4] = {1, 2, out_h, out_w};
  FloatTensor *conv_out = st_create(4, conv_shape);
  FloatTensor *bn_out = st_create(4, conv_shape);
  FloatTensor *mean = st_create(1, channel_shape);
  FloatTensor *var = st_create(1, channel_shape);

  if (conv_out == NULL || bn_out == NULL || mean == NULL || var == NULL) {
    fprintf(stderr, "Failed to allocate intermediate tensors.\n");
    st_destroy(var);
    st_destroy(mean);
    st_destroy(bn_out);
    st_destroy(conv_out);
    st_destroy(beta);
    st_destroy(gamma);
    st_destroy(weight);
    st_destroy(input);
    return 4;
  }

  if (!st_conv2d_nchw(input, weight, NULL, &conv_params, conv_out)) {
    fprintf(stderr, "Conv2D forward failed.\n");
    st_destroy(var);
    st_destroy(mean);
    st_destroy(bn_out);
    st_destroy(conv_out);
    st_destroy(beta);
    st_destroy(gamma);
    st_destroy(weight);
    st_destroy(input);
    return 5;
  }

  if (!st_batchnorm2d_forward_relu(conv_out, gamma, beta, 1e-5f, bn_out, mean,
                                   var)) {
    fprintf(stderr, "BatchNorm+ReLU forward failed.\n");
    st_destroy(var);
    st_destroy(mean);
    st_destroy(bn_out);
    st_destroy(conv_out);
    st_destroy(beta);
    st_destroy(gamma);
    st_destroy(weight);
    st_destroy(input);
    return 6;
  }

  const size_t pooled_shape[4] = {1, 2, 1, 1};
  FloatTensor *pooled = st_create(4, pooled_shape);
  if (pooled == NULL || !st_global_avgpool2d_nchw(bn_out, pooled)) {
    fprintf(stderr, "GlobalAvgPool forward failed.\n");
    st_destroy(pooled);
    st_destroy(var);
    st_destroy(mean);
    st_destroy(bn_out);
    st_destroy(conv_out);
    st_destroy(beta);
    st_destroy(gamma);
    st_destroy(weight);
    st_destroy(input);
    return 7;
  }

  puts("NN forward pass complete (Conv -> BatchNorm+ReLU -> GlobalAvgPool)");
  puts("Output scores [N=1, C=2, H=1, W=1]:");
  for (size_t c = 0; c < 2; ++c) {
    const size_t idx[4] = {0, c, 0, 0};
    printf("  class_%zu: %.6f\n", c, st_get(pooled, idx));
  }

  st_destroy(pooled);
  st_destroy(var);
  st_destroy(mean);
  st_destroy(bn_out);
  st_destroy(conv_out);
  st_destroy(beta);
  st_destroy(gamma);
  st_destroy(weight);
  st_destroy(input);
  return 0;
}
