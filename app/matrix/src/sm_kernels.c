#include "sm_kernels.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpedantic"
#endif
#include <omp.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(USE_ACCELERATE)
#include <Accelerate/Accelerate.h>

static void sm_kernel_vvexpf(float *values, size_t length) {
  for (size_t offset = 0; offset < length;) {
    const size_t remaining = length - offset;
    const int count = (int)(remaining > (size_t)INT32_MAX
                                ? (size_t)INT32_MAX
                                : remaining);
    vvexpf(values + offset, values + offset, &count);
    offset += (size_t)count;
  }
}

static void sm_kernel_vvtanhf(float *values, size_t length) {
  for (size_t offset = 0; offset < length;) {
    const size_t remaining = length - offset;
    const int count = (int)(remaining > (size_t)INT32_MAX
                                ? (size_t)INT32_MAX
                                : remaining);
    vvtanhf(values + offset, values + offset, &count);
    offset += (size_t)count;
  }
}
#endif

static bool sm_kernel_valid_buffer(const float *values, size_t length) {
  return values != NULL && length > 0;
}

#if defined(__clang__)
__attribute__((noinline, optnone))
#elif defined(__GNUC__)
__attribute__((noinline, optimize("no-fast-math")))
#endif
static bool sm_kernel_is_nan_bits(const float *value) {
  uint32_t bits = 0;
  memcpy(&bits, value, sizeof(bits));
  return (bits & 0x7f800000u) == 0x7f800000u &&
         (bits & 0x007fffffu) != 0;
}

bool sm_kernel_apply_sigmoid(float *values, size_t length) {
  if (!sm_kernel_valid_buffer(values, length)) {
    return false;
  }

#if defined(USE_ACCELERATE)
  vDSP_vneg(values, 1, values, 1, (vDSP_Length)length);
  sm_kernel_vvexpf(values, length);
  const float one = 1.0f;
  vDSP_vsadd(values, 1, &one, values, 1, (vDSP_Length)length);
  for (size_t i = 0; i < length; ++i) {
    values[i] = 1.0f / values[i];
  }
#else
#pragma omp parallel for simd schedule(static) if (length > 10000)
  for (size_t i = 0; i < length; ++i) {
    values[i] = 1.0f / (1.0f + expf(-values[i]));
  }
#endif
  return true;
}

bool sm_kernel_apply_tanh(float *values, size_t length) {
  if (!sm_kernel_valid_buffer(values, length)) {
    return false;
  }

#if defined(USE_ACCELERATE)
  sm_kernel_vvtanhf(values, length);
#else
#pragma omp parallel for simd schedule(static) if (length > 10000)
  for (size_t i = 0; i < length; ++i) {
    values[i] = tanhf(values[i]);
  }
#endif
  return true;
}

bool sm_kernel_apply_gelu_tanh(float *values, size_t length) {
  if (!sm_kernel_valid_buffer(values, length)) {
    return false;
  }

  const float scale = 0.7978845608f;
  const float cubic = 0.044715f;
#if defined(USE_ACCELERATE)
  enum { BLOCK = 256 };
  const size_t blocks = (length + BLOCK - 1) / BLOCK;
#pragma omp parallel for schedule(static) if (length > 10000)
  for (size_t block = 0; block < blocks; ++block) {
    float original[BLOCK];
    float transformed[BLOCK];
    const size_t offset = block * BLOCK;
    const size_t count = length - offset < BLOCK ? length - offset : BLOCK;
    for (size_t j = 0; j < count; ++j) {
      const float x = values[offset + j];
      original[j] = x;
      transformed[j] = scale * (x + cubic * x * x * x);
    }
    sm_kernel_vvtanhf(transformed, count);
    for (size_t j = 0; j < count; ++j) {
      values[offset + j] =
          0.5f * original[j] * (1.0f + transformed[j]);
    }
  }
#else
#pragma omp parallel for simd schedule(static) if (length > 10000)
  for (size_t i = 0; i < length; ++i) {
    const float x = values[i];
    values[i] = 0.5f * x * (1.0f + tanhf(scale * (x + cubic * x * x * x)));
  }
#endif
  return true;
}

bool sm_kernel_apply_gelu_erf(float *values, size_t length) {
  if (!sm_kernel_valid_buffer(values, length)) {
    return false;
  }

#pragma omp parallel for simd schedule(static) if (length > 10000)
  for (size_t i = 0; i < length; ++i) {
    const float x = values[i];
    values[i] = 0.5f * x * (1.0f + erff(x * 0.7071067812f));
  }
  return true;
}

bool sm_kernel_apply_activation(float *values, size_t length,
                                SmActivation activation) {
  if (activation == SM_ACT_NONE) {
    return values != NULL && length > 0;
  }
  if (!sm_kernel_valid_buffer(values, length)) {
    return false;
  }

  switch (activation) {
    case SM_ACT_RELU:
#if defined(USE_ACCELERATE)
      {
        const float zero = 0.0f;
        vDSP_vthr(values, 1, &zero, values, 1, (vDSP_Length)length);
      }
#else
#pragma omp parallel for simd schedule(static) if (length > 10000)
      for (size_t i = 0; i < length; ++i) {
        if (values[i] < 0.0f) {
          values[i] = 0.0f;
        }
      }
#endif
      return true;
    case SM_ACT_SIGMOID:
      return sm_kernel_apply_sigmoid(values, length);
    case SM_ACT_TANH:
      return sm_kernel_apply_tanh(values, length);
    case SM_ACT_NONE:
      return true;
  }
  return false;
}

bool sm_kernel_apply_softmax(float *values, size_t rows, size_t cols) {
  if (values == NULL || rows == 0 || cols == 0 ||
      rows > SIZE_MAX / cols) {
    return false;
  }

#if defined(USE_ACCELERATE)
  for (size_t row = 0; row < rows; ++row) {
    float *current = values + row * cols;
    float maximum = 0.0f;
    vDSP_maxv(current, 1, &maximum, (vDSP_Length)cols);
    const float negative_maximum = -maximum;
    vDSP_vsadd(current, 1, &negative_maximum, current, 1, (vDSP_Length)cols);
    sm_kernel_vvexpf(current, cols);
    float sum = 0.0f;
    vDSP_sve(current, 1, &sum, (vDSP_Length)cols);
    if (sum == 0.0f) {
      continue;
    }
    const float inverse = 1.0f / sum;
    vDSP_vsmul(current, 1, &inverse, current, 1, (vDSP_Length)cols);
  }
#else
#pragma omp parallel for schedule(static) if (rows > 1 && rows * cols > 10000)
  for (size_t row = 0; row < rows; ++row) {
    float *current = values + row * cols;
    float maximum = current[0];
    for (size_t col = 1; col < cols; ++col) {
      if (current[col] > maximum) {
        maximum = current[col];
      }
    }

    float sum = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
      current[col] = expf(current[col] - maximum);
      sum += current[col];
    }
    if (sum == 0.0f) {
      continue;
    }
    for (size_t col = 0; col < cols; ++col) {
      current[col] /= sum;
    }
  }
#endif
  return true;
}

bool sm_kernel_add_bias_activation(float *values, const float *bias,
                                   size_t rows, size_t cols, size_t bias_rows,
                                   SmActivation activation) {
  if (values == NULL || bias == NULL || rows == 0 || cols == 0 ||
      rows > SIZE_MAX / cols || (bias_rows != 1 && bias_rows != rows) ||
      (activation != SM_ACT_NONE && activation != SM_ACT_RELU &&
       activation != SM_ACT_SIGMOID && activation != SM_ACT_TANH)) {
    return false;
  }

  const size_t count = rows * cols;
#if defined(USE_ACCELERATE)
  if (activation == SM_ACT_NONE) {
    for (size_t row = 0; row < rows; ++row) {
      const float *bias_row = bias + (bias_rows == 1 ? 0 : row * cols);
      vDSP_vadd(values + row * cols, 1, bias_row, 1, values + row * cols, 1,
                (vDSP_Length)cols);
    }
    return true;
  }
  if (activation == SM_ACT_RELU) {
    for (size_t row = 0; row < rows; ++row) {
      const float *bias_row = bias + (bias_rows == 1 ? 0 : row * cols);
      vDSP_vadd(values + row * cols, 1, bias_row, 1, values + row * cols, 1,
                (vDSP_Length)cols);
    }
    const float zero = 0.0f;
    vDSP_vthr(values, 1, &zero, values, 1, (vDSP_Length)count);
    return true;
  }

  enum { BLOCK = 256 };
  const size_t blocks_per_row = (cols + BLOCK - 1) / BLOCK;
  const size_t blocks = rows * blocks_per_row;
#pragma omp parallel for schedule(static) if (count > 65536)
  for (size_t block = 0; block < blocks; ++block) {
    float work[BLOCK];
    const size_t row = block / blocks_per_row;
    const size_t col_offset = (block % blocks_per_row) * BLOCK;
    const size_t length = cols - col_offset < BLOCK ? cols - col_offset : BLOCK;
    const float *bias_row = bias + (bias_rows == 1 ? 0 : row * cols);
    float *output = values + row * cols + col_offset;
    for (size_t j = 0; j < length; ++j) {
      work[j] = output[j] + bias_row[col_offset + j];
    }
    (void)sm_kernel_apply_activation(work, length, activation);
    memcpy(output, work, length * sizeof(float));
  }
#else
#pragma omp parallel for schedule(static) if (count > 65536)
  for (size_t i = 0; i < count; ++i) {
    const float bias_value = bias_rows == 1 ? bias[i % cols] : bias[i];
    if (activation == SM_ACT_RELU && sm_kernel_is_nan_bits(values + i)) {
      continue;
    }
    if (activation == SM_ACT_RELU && sm_kernel_is_nan_bits(&bias_value)) {
      values[i] = bias_value;
      continue;
    }
    float value = values[i] + bias_value;
    switch (activation) {
      case SM_ACT_NONE:
        break;
      case SM_ACT_RELU:
        if (value < 0.0f) {
          value = 0.0f;
        }
        break;
      case SM_ACT_SIGMOID:
        value = 1.0f / (1.0f + expf(-value));
        break;
      case SM_ACT_TANH:
        value = tanhf(value);
        break;
    }
    values[i] = value;
  }
#endif
  return true;
}

bool sm_kernel_relu_backward(const float *activation, float *grad,
                             size_t length) {
  if (activation == NULL || grad == NULL || length == 0) {
    return false;
  }
#pragma omp parallel for simd schedule(static) if (length > 10000)
  for (size_t i = 0; i < length; ++i) {
    if (activation[i] <= 0.0f) {
      grad[i] = 0.0f;
    }
  }
  return true;
}

bool sm_kernel_sigmoid_backward(const float *activation, float *grad,
                                size_t length) {
  if (activation == NULL || grad == NULL || length == 0) {
    return false;
  }
#pragma omp parallel for simd schedule(static) if (length > 10000)
  for (size_t i = 0; i < length; ++i) {
    const float sigmoid = activation[i];
    grad[i] *= sigmoid * (1.0f - sigmoid);
  }
  return true;
}

bool sm_kernel_tanh_backward(const float *activation, float *grad,
                             size_t length) {
  if (activation == NULL || grad == NULL || length == 0) {
    return false;
  }
#pragma omp parallel for simd schedule(static) if (length > 10000)
  for (size_t i = 0; i < length; ++i) {
    const float tanh_value = activation[i];
    grad[i] *= 1.0f - tanh_value * tanh_value;
  }
  return true;
}

bool sm_kernel_gelu_tanh_backward(const float *input, float *grad,
                                  size_t length) {
  if (input == NULL || grad == NULL || length == 0) {
    return false;
  }
  const float scale = 0.7978845608f;
  const float cubic = 0.044715f;
#pragma omp parallel for simd schedule(static) if (length > 10000)
  for (size_t i = 0; i < length; ++i) {
    const float x = input[i];
    const float u = scale * (x + cubic * x * x * x);
    const float t = tanhf(u);
    const float derivative =
        0.5f * (1.0f + t) +
        0.5f * x * (1.0f - t * t) * scale * (1.0f + 3.0f * cubic * x * x);
    grad[i] *= derivative;
  }
  return true;
}

bool sm_kernel_gelu_erf_backward(const float *input, float *grad,
                                 size_t length) {
  if (input == NULL || grad == NULL || length == 0) {
    return false;
  }
  const float inv_sqrt_two = 0.7071067812f;
  const float inv_sqrt_two_pi = 0.3989422804f;
#pragma omp parallel for simd schedule(static) if (length > 10000)
  for (size_t i = 0; i < length; ++i) {
    const float x = input[i];
    const float cdf = 0.5f * (1.0f + erff(x * inv_sqrt_two));
    const float pdf = inv_sqrt_two_pi * expf(-0.5f * x * x);
    grad[i] *= cdf + x * pdf;
  }
  return true;
}