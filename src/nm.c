#include "nm.h"
#include <float.h>
#include <log.h>
#include <math.h>
#include <omp.h>
#include <stddef.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define INIT_CAPACITY 100
#define EPSILON 1e-5

/*******************************/
/*      Define Environment     */
/*******************************/

#if defined(USE_ACCELERATE)
#define ACTIVE_LIB "Apple Accelerate"
#elif defined(USE_ACCELERATE_MPS)
#define ACTIVE_LIB "Metal Performance Shaders"
#elif defined(USE_OPENBLAS)
#define ACTIVE_LIB "OpenBLAS"
#else
#if defined(__ARM_NEON)
#define ACTIVE_LIB "No BLAS, ARM NEON"
#else
#define ACTIVE_LIB "No BLAS"
#endif
#endif

#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
#define BLASINT int
#include "sm_mps.h"
#include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
#define BLASINT int
#include <cblas.h>
#include <lapacke.h>
#endif

// Block size used for cache-optimized transpose operations
#define BLOCK_SIZE 64

/*******************************/
/*          Functions.         */
/*******************************/

const char *nm_active_library(void) { return ACTIVE_LIB; }

void nm_apply_relu(FloatMatrix *mat) {
  if (!mat || !mat->values)
    return;

  size_t n = mat->rows * mat->cols;

#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)

  float zero = 0.0f;
  vDSP_vthres(mat->values, 1, &zero, mat->values, 1, n);

#elif defined(__ARM_NEON)
  float *data = mat->values;
  float32x4_t zeros = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t vals = vld1q_f32(&data[i]);
    float32x4_t relu = vmaxq_f32(vals, zeros);
    vst1q_f32(&data[i], relu);
  }
  for (; i < n; ++i) {
    data[i] = fmaxf(data[i], 0.0f);
  }

#else
  if (n > 1000000) {
#pragma omp parallel for simd
  } else {
#pragma omp simd
  }
  for (size_t i = 0; i < n; ++i) {
    float x = mat->values[i];
    mat->values[i] = x > 0.0f ? x : 0.0f;
  }
#endif
}

void nm_d_relu(const FloatMatrix *activation, FloatMatrix *grad_output) {
  if (!activation || !grad_output || activation->rows != grad_output->rows ||
      activation->cols != grad_output->cols)
    return;

  size_t size = activation->rows * activation->cols;
  const float *a = activation->values;
  float *g = grad_output->values;

#if defined(__ARM_NEON)
  size_t i = 0;
  float32x4_t zero = vdupq_n_f32(0.0f);
  for (; i + 4 <= size; i += 4) {
    float32x4_t act = vld1q_f32(&a[i]);
    float32x4_t grad = vld1q_f32(&g[i]);
    uint32x4_t mask = vcgtq_f32(act, zero);
    float32x4_t result = vbslq_f32(mask, grad, zero);
    vst1q_f32(&g[i], result);
  }
  for (; i < size; ++i) {
    g[i] = (a[i] > 0.0f) ? g[i] : 0.0f;
  }

#elif defined(USE_ACCELERATE)
  for (size_t i = 0; i < size; ++i) {
    g[i] = (a[i] > 0.0f) ? g[i] : 0.0f;
  }

#else
#pragma omp parallel for simd
  for (size_t i = 0; i < size; ++i) {
    g[i] = (a[i] > 0.0f) ? g[i] : 0.0f;
  }
#endif
}

void nm_d_sigmoid(const FloatMatrix *activation, FloatMatrix *grad_output) {
  if (!activation || !grad_output || activation->rows != grad_output->rows ||
      activation->cols != grad_output->cols)
    return;

  size_t size = activation->rows * activation->cols;
  const float *a = activation->values;
  float *g = grad_output->values;

#if defined(__ARM_NEON)
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t sig = vld1q_f32(&a[i]);
    float32x4_t grad = vld1q_f32(&g[i]);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t ds = vmulq_f32(sig, vsubq_f32(one, sig)); // s * (1 - s)
    vst1q_f32(&g[i], vmulq_f32(ds, grad));
  }
  for (; i < size; ++i) {
    float s = a[i];
    g[i] *= s * (1.0f - s);
  }

#elif defined(USE_ACCELERATE)
  for (size_t i = 0; i < size; ++i) {
    float s = a[i];
    g[i] *= s * (1.0f - s);
  }

#else
#pragma omp parallel for simd
  for (size_t i = 0; i < size; ++i) {
    float s = a[i];
    g[i] *= s * (1.0f - s);
  }
#endif
}

void nm_d_tanh(const FloatMatrix *activation, FloatMatrix *grad_output) {
  if (!activation || !grad_output || activation->rows != grad_output->rows ||
      activation->cols != grad_output->cols)
    return;

  size_t size = activation->rows * activation->cols;
  const float *a = activation->values;
  float *g = grad_output->values;

#if defined(__ARM_NEON)
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t tanh_vals = vld1q_f32(&a[i]);
    float32x4_t grad_vals = vld1q_f32(&g[i]);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t square = vmulq_f32(tanh_vals, tanh_vals);
    float32x4_t dtanh = vsubq_f32(one, square); // 1 - tanh^2(x)
    float32x4_t result = vmulq_f32(grad_vals, dtanh);
    vst1q_f32(&g[i], result);
  }
  for (; i < size; ++i) {
    float t = a[i];
    g[i] *= (1.0f - t * t);
  }

#elif defined(USE_ACCELERATE)
  for (size_t i = 0; i < size; ++i) {
    float t = a[i];
    g[i] *= (1.0f - t * t);
  }

#else
#pragma omp parallel for simd
  for (size_t i = 0; i < size; ++i) {
    float t = a[i];
    g[i] *= (1.0f - t * t);
  }
#endif
}

void nm_apply_sigmoid(FloatMatrix *mat) {
  if (!mat || !mat->values)
    return;

  size_t size = mat->rows * mat->cols;
  float *data = mat->values;

#if defined(USE_ACCELERATE)
  vvexpf(data, data, &size); // now data[i] = exp(data[i])
  for (size_t i = 0; i < size; ++i) {
    data[i] = 1.0f / (1.0f + 1.0f / data[i]);
  }

#elif defined(__ARM_NEON)
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t x = vld1q_f32(&data[i]);
    float32x4_t neg_x = vnegq_f32(x);
    float vals[4];
    vst1q_f32(vals, neg_x);
    for (int j = 0; j < 4; ++j) {
      vals[j] = 1.0f / (1.0f + expf(vals[j]));
    }
    vst1q_f32(&data[i], vld1q_f32(vals));
  }
  for (; i < size; ++i) {
    data[i] = 1.0f / (1.0f + expf(-data[i]));
  }

#else
  if (size > 1000000) {
#pragma omp parallel for simd
  } else {
#pragma omp simd
  }
  for (size_t i = 0; i < size; ++i) {
    float x = data[i];
    data[i] = 1.0f / (1.0f + expf(-x));
  }
#endif
}

float nm_tanh(float x) { return tanhf(x); }

void nm_apply_tanh(FloatMatrix *mat) {
  if (!mat || !mat->values)
    return;

  size_t n = mat->rows * mat->cols;
  float *data = mat->values;

#if defined(USE_ACCELERATE)
  vvtanhf(data, data, &n);
#elif defined(__ARM_NEON)
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float vals[4];
    vst1q_f32(vals, vld1q_f32(&data[i]));
    for (int j = 0; j < 4; ++j) {
      vals[j] = tanhf(vals[j]);
    }
    vst1q_f32(&data[i], vld1q_f32(vals));
  }
  for (; i < n; ++i) {
    data[i] = tanhf(data[i]);
  }
#else
#pragma omp simd
  for (size_t i = 0; i < n; ++i) {
    data[i] = tanhf(data[i]);
  }
#endif
}

void nm_inplace_add_rowwise(FloatMatrix *mat, const FloatMatrix *row) {
  if (!mat || !row || row->rows != 1 || row->cols != mat->cols)
    return;

#if defined(USE_ACCELERATE) || defined(USE_ACCELERATE_MPS)
  for (size_t i = 0; i < mat->rows; ++i) {
    vDSP_vadd(&mat->values[i * mat->cols], 1, row->values, 1,
              &mat->values[i * mat->cols], 1, mat->cols);
  }
#elif defined(__ARM_NEON)
  size_t cols = mat->cols;
  // size_t cols_aligned = cols & ~3UL;

  for (size_t i = 0; i < mat->rows; ++i) {
    float *dst = &mat->values[i * cols];
    size_t j = 0;
    for (; j + 4 <= cols; j += 4) {
      float32x4_t dst_vec = vld1q_f32(&dst[j]);
      float32x4_t bias_vec = vld1q_f32(&row->values[j]);
      float32x4_t sum = vaddq_f32(dst_vec, bias_vec);
      vst1q_f32(&dst[j], sum);
    }
    for (; j < cols; ++j) {
      dst[j] += row->values[j];
    }
  }
#else
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (size_t i = 0; i < mat->rows; ++i) {
    size_t base = i * mat->cols;
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (size_t j = 0; j < mat->cols; ++j) {
      mat->values[base + j] += row->values[j];
    }
  }
#endif
}

FloatMatrix *nm_sum_rows(const FloatMatrix *mat) {
  if (!mat || !mat->values)
    return NULL;

  FloatMatrix *result = sm_create_zeros(1, mat->cols);
  if (!result)
    return NULL;

  float *out = result->values;
  const float *in = mat->values;

#if defined(__ARM_NEON)
  size_t cols = mat->cols;
  for (size_t j = 0; j + 4 <= cols; j += 4) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (size_t i = 0; i < mat->rows; ++i) {
      float32x4_t vals = vld1q_f32(&in[i * cols + j]);
      sum_vec = vaddq_f32(sum_vec, vals);
    }
    vst1q_f32(&out[j], sum_vec);
  }
  for (size_t j = (cols & ~3UL); j < cols; ++j) {
    for (size_t i = 0; i < mat->rows; ++i) {
      out[j] += in[i * cols + j];
    }
  }

#else
#pragma omp parallel for
  for (size_t j = 0; j < mat->cols; ++j) {
    float sum = 0.0f;
    for (size_t i = 0; i < mat->rows; ++i) {
      sum += in[i * mat->cols + j];
    }
    out[j] = sum;
  }
#endif

  return result;
}

FloatMatrix *nm_linear(const FloatMatrix *input, const FloatMatrix *weights,
                       const FloatMatrix *bias) {
  if (!input || !weights || !bias)
    return NULL;
  if (input->cols != weights->rows || bias->cols != weights->cols ||
      bias->rows != 1)
    return NULL;

  FloatMatrix *out = sm_multiply(input, weights);
  if (!out)
    return NULL;

  nm_inplace_add_rowwise(out, bias);

  return out;
}

float nm_cross_entropy_loss(const FloatMatrix *predicted,
                            const FloatMatrix *target) {
  if (!predicted || !target)
    return -1.0f;
  if (predicted->rows != target->rows || predicted->cols != target->cols)
    return -1.0f;

  size_t size = predicted->rows * predicted->cols;
  const float *p = predicted->values;
  const float *t = target->values;
  float loss = 0.0f;

#if defined(__ARM_NEON)
  float32x4_t sum_vec = vdupq_n_f32(0.0f);
  float32x4_t epsilon = vdupq_n_f32(1e-7f);
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t pred = vld1q_f32(&p[i]);
    // float32x4_t targ = vld1q_f32(&t[i]);
    pred = vmaxq_f32(pred, epsilon); // clamp to epsilon
    float vals[4];
    vst1q_f32(vals, pred);
    for (int j = 0; j < 4; ++j) {
      vals[j] = -t[i + j] * logf(vals[j]);
    }
    sum_vec = vaddq_f32(sum_vec, vld1q_f32(vals));
  }
  float buf[4];
  vst1q_f32(buf, sum_vec);
  loss = buf[0] + buf[1] + buf[2] + buf[3];
  for (; i < size; ++i) {
    float pi = fmaxf(p[i], 1e-7f);
    loss += -t[i] * logf(pi);
  }

#elif defined(USE_ACCELERATE)
  float *log_p = (float *)malloc(size * sizeof(float));
  float *safe_p = (float *)malloc(size * sizeof(float));
  if (!log_p || !safe_p) {
    free(log_p);
    free(safe_p);
    return -1.0f;
  }

  for (size_t i = 0; i < size; ++i) {
    safe_p[i] = fmaxf(p[i], 1e-7f);
  }
  vvlogf(log_p, safe_p, &size);

  float sum = 0.0f;
  vDSP_dotpr(t, 1, log_p, 1, &sum, size);
  loss = -sum;

  free(log_p);
  free(safe_p);

#else
#pragma omp parallel for reduction(+ : loss)
  for (size_t i = 0; i < size; ++i) {
    float pi = fmaxf(p[i], 1e-7f);
    loss += -t[i] * logf(pi);
  }
#endif

  return loss / (float)predicted->rows;
}

float nm_softmax_denominator(const float *vec, size_t len) {
  if (!vec || len == 0)
    return 0.0f;

  float max_val = vec[0];
  for (size_t i = 1; i < len; ++i) {
    if (vec[i] > max_val)
      max_val = vec[i];
  }

#if defined(USE_ACCELERATE)
  float *tmp = (float *)malloc(len * sizeof(float));
  if (!tmp)
    return 0.0f;
  float neg_max = -max_val;
  vDSP_vsadd(vec, 1, &neg_max, tmp, 1, len);
  vvexpf(tmp, tmp, &len);
  float sum = 0.0f;
  vDSP_sve(tmp, 1, &sum, len);
  free(tmp);
  return sum;

#elif defined(__ARM_NEON)
  float sum = 0.0f;
  float32x4_t v_sum = vdupq_n_f32(0.0f);
  float32x4_t v_max = vdupq_n_f32(max_val);
  size_t i = 0;
  for (; i + 4 <= len; i += 4) {
    float32x4_t v = vld1q_f32(&vec[i]);
    float vals[4];
    vst1q_f32(vals, vsubq_f32(v, v_max));
    for (int j = 0; j < 4; ++j) {
      vals[j] = expf(vals[j]);
    }
    v_sum = vaddq_f32(v_sum, vld1q_f32(vals));
  }
  float buf[4];
  vst1q_f32(buf, v_sum);
  sum = buf[0] + buf[1] + buf[2] + buf[3];

  for (; i < len; ++i) {
    sum += expf(vec[i] - max_val);
  }
  return sum;

#else
  float sum = 0.0f;
  for (size_t i = 0; i < len; ++i) {
    sum += expf(vec[i] - max_val);
  }
  return sum;
#endif
}

// Optimized softmax implementation (row-wise)
void nm_apply_softmax(FloatMatrix *mat) {
  if (!mat || !mat->values)
    return;

  for (size_t i = 0; i < mat->rows; ++i) {
    float *row = &mat->values[i * mat->cols];
    size_t cols = mat->cols;

    // Step 1: Find max value for numerical stability
    float max_val = row[0];
    for (size_t j = 1; j < cols; ++j) {
      if (row[j] > max_val)
        max_val = row[j];
    }

#if defined(USE_ACCELERATE)
    float minus_max = -max_val;
    vDSP_vsadd(row, 1, &minus_max, row, 1, cols);
    vvexpf(row, row, &cols);
    float sum = 0.0f;
    vDSP_sve(row, 1, &sum, cols);
    float inv_sum = 1.0f / sum;
    vDSP_vsmul(row, 1, &inv_sum, row, 1, cols);

#elif defined(__ARM_NEON)
    float sum = 0.0f;
    for (size_t j = 0; j < cols; ++j) {
      row[j] = expf(row[j] - max_val);
      sum += row[j];
    }
    float inv_sum = 1.0f / sum;
    float32x4_t v_inv_sum = vdupq_n_f32(inv_sum);
    size_t j = 0;
    for (; j + 4 <= cols; j += 4) {
      float32x4_t v = vld1q_f32(&row[j]);
      vst1q_f32(&row[j], vmulq_f32(v, v_inv_sum));
    }
    for (; j < cols; ++j) {
      row[j] *= inv_sum;
    }

#else
    float sum = 0.0f;
    for (size_t j = 0; j < cols; ++j) {
      row[j] = expf(row[j] - max_val);
      sum += row[j];
    }
    for (size_t j = 0; j < cols; ++j) {
      row[j] /= sum;
    }
#endif
  }
}

float nm_mse_loss(const FloatMatrix *predicted, const FloatMatrix *target) {
  if (!predicted || !target)
    return -1.0f;
  if (predicted->rows != target->rows || predicted->cols != target->cols)
    return -1.0f;

  size_t size = predicted->rows * predicted->cols;
  const float *p = predicted->values;
  const float *t = target->values;
  float loss = 0.0f;

#if defined(__ARM_NEON)
  float32x4_t sum_vec = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t a = vld1q_f32(&p[i]);
    float32x4_t b = vld1q_f32(&t[i]);
    float32x4_t diff = vsubq_f32(a, b);
    float32x4_t sq = vmulq_f32(diff, diff);
    sum_vec = vaddq_f32(sum_vec, sq);
  }
  float buf[4];
  vst1q_f32(buf, sum_vec);
  loss = buf[0] + buf[1] + buf[2] + buf[3];
  for (; i < size; ++i) {
    float diff = p[i] - t[i];
    loss += diff * diff;
  }

#elif defined(USE_ACCELERATE)
  float *diff = (float *)malloc(size * sizeof(float));
  if (!diff)
    return -1.0f;
  vDSP_vsub(t, 1, p, 1, diff, 1, size);
  vDSP_svesq(diff, 1, &loss, size);
  free(diff);

#else
#pragma omp parallel for reduction(+ : loss)
  for (size_t i = 0; i < size; ++i) {
    float diff = p[i] - t[i];
    loss += diff * diff;
  }
#endif

  return loss / (float)size;
}

void nm_d_softmax_crossentropy(const FloatMatrix *predicted,
                               const FloatMatrix *target,
                               FloatMatrix *grad_output) {
  if (!predicted || !target || !grad_output)
    return;
  if (predicted->rows != target->rows || predicted->cols != target->cols ||
      predicted->rows != grad_output->rows ||
      predicted->cols != grad_output->cols)
    return;

  size_t size = predicted->rows * predicted->cols;
  const float *p = predicted->values;
  const float *t = target->values;
  float *g = grad_output->values;

#if defined(__ARM_NEON)
  size_t i = 0;
  for (; i + 4 <= size; i += 4) {
    float32x4_t v_p = vld1q_f32(&p[i]);
    float32x4_t v_t = vld1q_f32(&t[i]);
    vst1q_f32(&g[i], vsubq_f32(v_p, v_t));
  }
  for (; i < size; ++i) {
    g[i] = p[i] - t[i];
  }

#elif defined(USE_ACCELERATE)
  vDSP_vsub(t, 1, p, 1, g, 1, size);
  for (size_t i = 0; i < size; ++i) {
    g[i] = -g[i];
  }

#else
#pragma omp parallel for simd
  for (size_t i = 0; i < size; ++i) {
    g[i] = p[i] - t[i];
  }
#endif
}

FloatMatrix *dense_forward(DenseLayer *layer, const FloatMatrix *input) {
  if (!layer || !input || !layer->weights || !layer->bias || !layer->activation)
    return NULL;

  FloatMatrix *z = nm_linear(input, layer->weights, layer->bias);
  if (!z)
    return NULL;

  layer->activation(z);
  return z;
}

void dense_backward(DenseLayer *layer, const FloatMatrix *input,
                    const FloatMatrix *activation, FloatMatrix *grad_output,
                    float learning_rate) {
  if (!layer || !input || !activation || !grad_output || !layer->weights ||
      !layer->bias || !layer->activation_derivative)
    return;

  // Ableitung Aktivierung anwenden
  if (layer->activation_derivative != NULL) {
    layer->activation_derivative(activation, grad_output);
  }

  // Gradienten berechnen
  FloatMatrix *input_T = sm_transpose(input);
  FloatMatrix *grad_weights = sm_multiply(input_T, grad_output);
  sm_destroy(input_T);

  // Bias-Gradient = Zeilensumme von grad_output
  FloatMatrix *grad_bias = nm_sum_rows(grad_output);

// Update
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < layer->weights->rows; ++i) {
    for (size_t j = 0; j < layer->weights->cols; ++j) {
      layer->weights->values[i * layer->weights->cols + j] -=
          learning_rate * grad_weights->values[i * grad_weights->cols + j];
    }
  }

  for (size_t j = 0; j < layer->bias->cols; ++j) {
    layer->bias->values[j] -= learning_rate * grad_bias->values[j];
  }

  sm_destroy(grad_weights);
  sm_destroy(grad_bias);
}

void train_one_epoch(
    NeuralNetwork *net,
    const FloatMatrix *X,      // Input:  (num_samples x input_dim)
    const FloatMatrix *y_true, // Target: (num_samples x num_classes)
    size_t batch_size, float learning_rate) {
  size_t num_samples = X->rows;
  size_t num_batches = (num_samples + batch_size - 1) / batch_size;

  for (size_t batch = 0; batch < num_batches; ++batch) {
    size_t start = batch * batch_size;
    size_t end =
        (start + batch_size < num_samples) ? (start + batch_size) : num_samples;
    // size_t actual_batch_size = end - start;

    // Slice input and target batch
    FloatMatrix *X_batch = sm_slice_rows(X, start, end);
    FloatMatrix *y_batch = sm_slice_rows(y_true, start, end);

    // Forward pass
    FloatMatrix **activations =
        (FloatMatrix **)malloc((net->num_layers + 1) * sizeof(FloatMatrix *));
    activations[0] = X_batch;
    for (size_t i = 0; i < net->num_layers; ++i) {
      activations[i + 1] = dense_forward(&net->layers[i], activations[i]);
    }

    // Output layer error (Softmax + CrossEntropy loss assumed)
    FloatMatrix *grad = sm_clone(activations[net->num_layers]);
    nm_d_softmax_crossentropy(activations[net->num_layers], y_batch, grad);

    // Backward pass
    for (ssize_t i = net->num_layers - 1; i >= 0; --i) {
      dense_backward(&net->layers[i],
                     activations[i],     // input to this layer
                     activations[i + 1], // output of this layer
                     grad, learning_rate);

      if (i > 0) {
        FloatMatrix *W_T = sm_transpose(net->layers[i].weights);
        FloatMatrix *new_grad = sm_multiply(grad, W_T);
        sm_destroy(W_T);
        sm_destroy(grad);
        grad = new_grad;
      }
    }

    // Cleanup
    sm_destroy(grad);
    for (size_t i = 1; i <= net->num_layers; ++i) {
      sm_destroy(activations[i]);
    }
    free(activations);
    sm_destroy(X_batch);
    sm_destroy(y_batch);
  }
}

//

// Predict function: Forward pass through all layers of the network
FloatMatrix *predict(const NeuralNetwork *net, const FloatMatrix *input) {
  if (!net || !input || net->num_layers == 0)
    return NULL;

  FloatMatrix *current = sm_clone(input);
  for (size_t i = 0; i < net->num_layers; ++i) {
    FloatMatrix *next = dense_forward(&net->layers[i], current);
    sm_destroy(current);
    current = next;
  }
  return current; // output layer result
}

FloatMatrix *nm_argmax_rowwise(const FloatMatrix *mat) {
  if (!mat || !mat->values)
    return NULL;

  FloatMatrix *result = sm_create(mat->rows, 1);
  if (!result)
    return NULL;

#pragma omp parallel for
  for (size_t i = 0; i < mat->rows; ++i) {
    float max_val = mat->values[i * mat->cols];
    size_t max_idx = 0;
    for (size_t j = 1; j < mat->cols; ++j) {
      float val = mat->values[i * mat->cols + j];
      if (val > max_val) {
        max_val = val;
        max_idx = j;
      }
    }
    result->values[i] = (float)max_idx;
  }

  return result;
}

//
void train(NeuralNetwork *net, const FloatMatrix *X, const FloatMatrix *Y,
           size_t batch_size, float learning_rate, size_t epochs,
           bool verbose) {
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    train_one_epoch(net, X, Y, batch_size, learning_rate);

    if (verbose) {
      FloatMatrix *out = predict(net, X);
      float loss = nm_cross_entropy_loss(out, Y);
      float acc = accuracy_score(out, Y);
      log_info("Epoch %zu/%zu - Loss: %.5f - Accuracy: %.2f%%", epoch + 1,
               epochs, loss, acc * 100.0f);
      sm_destroy(out);
    }
  }
}

float accuracy_score(const FloatMatrix *predicted, const FloatMatrix *target) {
  if (!predicted || !target || predicted->rows != target->rows ||
      predicted->cols != target->cols)
    return -1.0f;

  FloatMatrix *pred_labels = nm_argmax_rowwise(predicted);
  FloatMatrix *true_labels = nm_argmax_rowwise(target);

  if (!pred_labels || !true_labels)
    return -1.0f;

  size_t correct = 0;
  for (size_t i = 0; i < predicted->rows; ++i) {
    if ((int)sm_get(pred_labels, i, 0) == (int)sm_get(true_labels, i, 0)) {
      ++correct;
    }
  }

  sm_destroy(pred_labels);
  sm_destroy(true_labels);

  return (float)correct / predicted->rows;
}
// Binary save/load for network weights
bool save_network_bin(const char *path, const NeuralNetwork *net) {
  FILE *f = fopen(path, "wb");
  if (!f || !net) return false;

  fwrite(&net->num_layers, sizeof(size_t), 1, f);
  for (size_t l = 0; l < net->num_layers; ++l) {
    DenseLayer *layer = &net->layers[l];
    size_t w_rows = layer->weights->rows;
    size_t w_cols = layer->weights->cols;
    fwrite(&w_rows, sizeof(size_t), 1, f);
    fwrite(&w_cols, sizeof(size_t), 1, f);
    fwrite(layer->weights->values, sizeof(float), w_rows * w_cols, f);
    fwrite(layer->bias->values, sizeof(float), layer->bias->cols, f);
  }

  fclose(f);
  return true;
}

bool load_network_bin(const char *path, NeuralNetwork *net) {
  FILE *f = fopen(path, "rb");
  if (!f || !net) return false;

  fread(&net->num_layers, sizeof(size_t), 1, f);
  net->layers = calloc(net->num_layers, sizeof(DenseLayer));
  if (!net->layers) return false;

  for (size_t l = 0; l < net->num_layers; ++l) {
    DenseLayer *layer = &net->layers[l];
    size_t w_rows, w_cols;
    fread(&w_rows, sizeof(size_t), 1, f);
    fread(&w_cols, sizeof(size_t), 1, f);
    layer->weights = sm_create(w_rows, w_cols);
    layer->bias = sm_create_zeros(1, w_cols);
    fread(layer->weights->values, sizeof(float), w_rows * w_cols, f);
    fread(layer->bias->values, sizeof(float), w_cols, f);

    // Assign default activation functions (assume ReLU for now)
    layer->activation = nm_apply_relu;
    layer->activation_derivative = nm_d_relu;
  }

  fclose(f);
  return true;
}