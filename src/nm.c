#include "nm.h"
#include <float.h>
#include <math.h>
#include <omp.h>

void nm_apply_relu(FloatMatrix *mat) {
  if (!mat)
    return;
  size_t size = mat->rows * mat->cols;

  if (size > 1000000) {
#pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
      float x = mat->values[i];
      mat->values[i] = x > 0.0f ? x : 0.0f;
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < size; ++i) {
      float x = mat->values[i];
      mat->values[i] = x > 0.0f ? x : 0.0f;
    }
  }
}

void nm_apply_sigmoid(FloatMatrix *mat) {
  size_t size = mat->rows * mat->cols;

  if (size > 1000000) {
#pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
      float x = mat->values[i];
      mat->values[i] = 1.0f / (1.0f + expf(-x));
    }
  } else {
#pragma omp simd
    for (size_t i = 0; i < size; ++i) {
      float x = mat->values[i];
      mat->values[i] = 1.0f / (1.0f + expf(-x));
    }
  }
}

float nm_tanh(float x) { return tanhf(x); }

void nm_apply_tanh(FloatMatrix *mat) {
  for (size_t i = 0; i < mat->rows * mat->cols; ++i) {
    mat->values[i] = nm_tanh(mat->values[i]);
  }
}

FloatMatrix *nm_linear(const FloatMatrix *input, const FloatMatrix *weights,
                       const FloatMatrix *bias) {
  if (!input || !weights || !bias)
    return NULL;
  if (input->cols != weights->rows || bias->cols != weights->cols ||
      bias->rows != 1)
    return NULL;

  FloatMatrix *out = sm_create(input->rows, weights->cols);
  if (!out)
    return NULL;

  for (size_t i = 0; i < input->rows; ++i) {
    for (size_t j = 0; j < weights->cols; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < input->cols; ++k) {
        sum += sm_get(input, i, k) * sm_get(weights, k, j);
      }
      sum += sm_get(bias, 0, j);
      sm_set(out, i, j, sum);
    }
  }
  return out;
}

float nm_cross_entropy_loss(const FloatMatrix *predicted,
                            const FloatMatrix *target) {
  if (!predicted || !target)
    return -1.0f;
  if (predicted->rows != target->rows || predicted->cols != target->cols)
    return -1.0f;

  float loss = 0.0f;
  for (size_t i = 0; i < predicted->rows * predicted->cols; ++i) {
    float p = predicted->values[i];
    float t = target->values[i];
    // numerische Stabilisierung: p = max(p, ε)
    if (p < 1e-7f)
      p = 1e-7f;
    loss -= t * logf(p);
  }
  return loss / (predicted->rows * predicted->cols);
}
// Softmax denominator helper
float nm_softmax_denominator(const float *vec, size_t len) {
  float sum = 0.0f;
  for (size_t i = 0; i < len; ++i) {
    sum += expf(vec[i]);
  }
  return sum;
}

// Applies softmax row-wise to a FloatMatrix
void nm_apply_softmax(FloatMatrix *mat) {
  for (size_t i = 0; i < mat->rows; ++i) {
    // Find max value in row for numerical stability
    float max_val = mat->values[i * mat->cols];
    for (size_t j = 1; j < mat->cols; ++j) {
      float val = mat->values[i * mat->cols + j];
      if (val > max_val)
        max_val = val;
    }

    float sum = 0.0f;
    // Compute exp(x - max) for stability and sum
    for (size_t j = 0; j < mat->cols; ++j) {
      float expval = expf(mat->values[i * mat->cols + j] - max_val);
      mat->values[i * mat->cols + j] = expval;
      sum += expval;
    }
    // Normalize
    for (size_t j = 0; j < mat->cols; ++j) {
      mat->values[i * mat->cols + j] /= sum;
    }
  }
}

float nm_mse_loss(const FloatMatrix *predicted, const FloatMatrix *target) {
  if (!predicted || !target)
    return -1.0f;
  if (predicted->rows != target->rows || predicted->cols != target->cols)
    return -1.0f;

  float loss = 0.0f;
  for (size_t i = 0; i < predicted->rows * predicted->cols; ++i) {
    float diff = predicted->values[i] - target->values[i];
    loss += diff * diff;
  }
  return loss / (predicted->rows * predicted->cols);
}