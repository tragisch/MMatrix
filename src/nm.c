#include "nm.h"
#include <float.h>
#include <math.h>
#include <omp.h>

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
#define ACTIVE_LIB "BLAS, ARM NEON"
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
  if (size > 1000000) {
#pragma omp parallel for simd
  } else {
#pragma omp simd
  }
  for (size_t i = 0; i < size; ++i) {
    float x = mat->values[i];
    mat->values[i] = x > 0.0f ? x : 0.0f;
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