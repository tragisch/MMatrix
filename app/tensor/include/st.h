/**
 * @file st.h
 * @brief Public core API for float tensor creation, views, element-wise ops, and utilities.
 *
 * This header intentionally exposes an opaque @ref FloatTensor type for external users.
 * Internal layout is available only for internal build targets via
 * `ST_INTERNAL_TENSOR_LAYOUT`.
 */

/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef ST_H
#define ST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ST_MAX_DIMS 8

typedef struct StBuffer StBuffer;

/** @brief Element type used by tensor storage. */
typedef enum StDtype {
  ST_DTYPE_F32 = 0,   /* IEEE 754 binary32  (4 bytes, default) */
  ST_DTYPE_BF16 = 1,  /* bfloat16           (2 bytes)          */
} StDtype;

/**
 * @brief Return storage size in bytes for a tensor element type.
 * @param dtype Element type.
 * @return 4 for @ref ST_DTYPE_F32, 2 for @ref ST_DTYPE_BF16.
 */
static inline size_t st_dtype_size(StDtype dtype) {
  switch (dtype) {
    case ST_DTYPE_BF16:
      return 2;
    case ST_DTYPE_F32:
    default:
      return 4;
  }
}

/** @brief Logical tensor layout hint. */
typedef enum StLayout {
  ST_LAYOUT_CONTIGUOUS = 0,
  ST_LAYOUT_NCHW = 1,
  ST_LAYOUT_NHWC = 2,
  ST_LAYOUT_TBF = 3,  // [time, batch, feature]
  ST_LAYOUT_BTF = 4,  // [batch, time, feature]
} StLayout;

typedef struct FloatMatrix FloatMatrix;

/** @brief Opaque tensor handle used by all public tensor APIs. */
typedef struct FloatTensor FloatTensor;

#ifdef ST_INTERNAL_TENSOR_LAYOUT
struct FloatTensor {
  size_t ndim;
  size_t shape[ST_MAX_DIMS];
  ptrdiff_t strides[ST_MAX_DIMS];
  size_t numel;
  size_t capacity;
  float *values;
  bool owns_data;
  StLayout layout;
  StDtype dtype;     /* element type: ST_DTYPE_F32 (default) | ST_DTYPE_BF16 */

  /* ---- Buffer-backed storage (Phase 1) ---- */
  StBuffer *buf;                    // ref-counted backing storage
  size_t view_offset;               // element offset into buf->data
  struct FloatTensor *view_src;     // base tensor for views; NULL if owner
  void *extra;                      // backend-specific hook (pipeline cache, …)
  void (*extra_free)(void *);       // destructor for extra (NULL = no-op)
};
#endif

/**
 * @brief Compute default row-major strides for a shape.
 * @param ndim Number of dimensions.
 * @param shape Input shape array of length @p ndim.
 * @param out_strides Output stride array of length @p ndim.
 * @retval true Success.
 * @retval false Invalid input or overflow.
 */
bool st_compute_default_strides(size_t ndim, const size_t *shape,
                                ptrdiff_t *out_strides);

/**
 * @brief Compute number of elements from a shape.
 * @param ndim Number of dimensions.
 * @param shape Input shape array of length @p ndim.
 * @param out_numel Output element count.
 * @retval true Success.
 * @retval false Invalid input or overflow.
 */
bool st_numel_from_shape(size_t ndim, const size_t *shape, size_t *out_numel);

/**
 * @brief Create a contiguous f32 tensor with zero-initialized storage.
 * @param ndim Number of dimensions.
 * @param shape Shape array of length @p ndim.
 * @return New tensor on success, or NULL on error.
 */
FloatTensor *st_create(size_t ndim, const size_t *shape);

/**
 * @brief Create a contiguous bf16 tensor with zero-initialized storage.
 * @param ndim Number of dimensions.
 * @param shape Shape array of length @p ndim.
 * @return New tensor on success, or NULL on error.
 */
FloatTensor *st_create_bf16(size_t ndim, const size_t *shape);

/**
 * @brief Wrap existing f32 data in a tensor object.
 * @param ndim Number of dimensions.
 * @param shape Shape array of length @p ndim.
 * @param data Pointer to existing f32 storage.
 * @param capacity Capacity in float elements of @p data.
 * @param take_ownership When true, tensor destructor will free @p data.
 * @return New tensor on success, or NULL on error.
 */
FloatTensor *st_create_with_data(size_t ndim, const size_t *shape, float *data,
                                 size_t capacity, bool take_ownership);

/**
 * @brief Create deep copy of a tensor.
 * @param src Source tensor.
 * @return New cloned tensor on success, or NULL on error.
 */
FloatTensor *st_clone(const FloatTensor *src);

/**
 * @brief Convert tensor to f32 storage.
 * @param src Source tensor.
 * @return New f32 tensor, or NULL on error.
 */
FloatTensor *st_to_f32(const FloatTensor *src);

/**
 * @brief Convert tensor to bf16 storage.
 * @param src Source tensor.
 * @return New bf16 tensor, or NULL on error.
 */
FloatTensor *st_to_bf16(const FloatTensor *src);

/**
 * @brief Create view with custom shape, strides, and offset.
 * @param base Base tensor to view into.
 * @param ndim Number of dimensions in the view.
 * @param shape View shape array.
 * @param strides View stride array.
 * @param offset_elements Element offset from base start.
 * @return New view tensor, or NULL on error.
 */
FloatTensor *st_view(FloatTensor *base, size_t ndim, const size_t *shape,
                     const ptrdiff_t *strides, size_t offset_elements);

/**
 * @brief Check whether tensor has default contiguous row-major strides.
 * @param tensor Tensor to inspect.
 * @retval true Tensor is contiguous.
 * @retval false Tensor is non-contiguous or invalid.
 */
bool st_is_contiguous(const FloatTensor *tensor);

/**
 * @brief Reshape tensor in-place.
 * @param tensor Tensor to reshape.
 * @param new_ndim Number of new dimensions.
 * @param new_shape New shape array.
 * @retval true Success.
 * @retval false Invalid reshape (including numel mismatch).
 */
bool st_reshape(FloatTensor *tensor, size_t new_ndim, const size_t *new_shape);

/**
 * @brief Create a permuted view without copying storage.
 * @param base Base tensor.
 * @param perm Permutation array of length tensor ndim.
 * @return New permuted view, or NULL on error.
 */
FloatTensor *st_permute_view(FloatTensor *base, const size_t *perm);

/**
 * @brief Read one scalar from a tensor at multi-index.
 * @param tensor Source tensor.
 * @param indices Multi-index array of length tensor ndim.
 * @return Scalar value, or 0.0f on invalid input.
 */
float st_get(const FloatTensor *tensor, const size_t *indices);

/**
 * @brief Write one scalar into a tensor at multi-index.
 * @param tensor Destination tensor.
 * @param indices Multi-index array of length tensor ndim.
 * @param value Scalar value to write.
 * @retval true Success.
 * @retval false Invalid input or index.
 */
bool st_set(FloatTensor *tensor, const size_t *indices, float value);

/**
 * @brief Expose 2D contiguous f32 tensor as @ref FloatMatrix view.
 * @param tensor Source tensor.
 * @param out_view Output matrix view.
 * @retval true Success.
 * @retval false Tensor is incompatible.
 */
bool st_as_sm_view(const FloatTensor *tensor, FloatMatrix *out_view);

/**
 * @brief Destroy tensor handle and release associated storage/resources.
 * @param tensor Tensor to destroy (NULL-safe).
 */
void st_destroy(FloatTensor *tensor);

/**
 * @brief Wait for pending async GPU writes to complete.
 * @param tensor Tensor to synchronize (NULL-safe).
 */
void st_tensor_sync(FloatTensor *tensor);

// ---- Read-only tensor metadata/data accessors (opaque-ready API) ----

/** @brief Return number of dimensions, or 0 for NULL tensor. */
size_t st_tensor_ndim(const FloatTensor *tensor);

/** @brief Return shape pointer, or NULL for NULL tensor. */
const size_t *st_tensor_shape(const FloatTensor *tensor);

/** @brief Return logical element count, or 0 for NULL tensor. */
size_t st_tensor_numel(const FloatTensor *tensor);

/** @brief Return dtype, defaults to @ref ST_DTYPE_F32 for NULL tensor. */
StDtype st_tensor_dtype(const FloatTensor *tensor);

/**
 * @brief Return raw const data pointer.
 * @details For bf16 tensors this points to packed bf16 storage.
 */
const float *st_tensor_data(const FloatTensor *tensor);

/**
 * @brief Return raw mutable data pointer.
 * @details For bf16 tensors this points to packed bf16 storage.
 */
float *st_tensor_mutable_data(FloatTensor *tensor);

// ---- Element-wise in-place operations ----

/**
 * @brief In-place add: @p a[i] += @p b[i].
 * @retval true Success.
 * @retval false Invalid input or shape mismatch.
 */
bool st_inplace_add(FloatTensor *a, const FloatTensor *b);

/** @brief In-place subtract: @p a[i] -= @p b[i]. */
bool st_inplace_sub(FloatTensor *a, const FloatTensor *b);

/** @brief In-place scale: @p t[i] *= @p scalar. */
bool st_inplace_scale(FloatTensor *t, float scalar);

/** @brief In-place Hadamard product: @p a[i] *= @p b[i]. */
bool st_inplace_elementwise_multiply(FloatTensor *a, const FloatTensor *b);

/** @brief Fill all tensor elements with one scalar value. */
bool st_fill(FloatTensor *t, float value);

// ---- Activation functions on tensors ----

/** @brief Apply ReLU in-place: `t[i] = max(0, t[i])`. */
bool st_apply_relu(FloatTensor *t);

/** @brief Apply ReLU backward mask in-place to @p grad. */
bool st_apply_relu_backward(const FloatTensor *activation, FloatTensor *grad);

// ---- Reduction ----

/**
 * @brief Reduce tensor by summing over selected axes.
 * @param t Input tensor.
 * @param axes Array of axis indices to reduce.
 * @param num_axes Number of axes in @p axes.
 * @return Reduced tensor on success, or NULL on error.
 */
FloatTensor *st_sum_axes(const FloatTensor *t, const size_t *axes,
                         size_t num_axes);

// ---- Padding ----

/**
 * @brief Pad NCHW tensor spatially.
 * @param input Input tensor with shape `[N, C, H, W]`.
 * @param pad_h Symmetric padding in height dimension.
 * @param pad_w Symmetric padding in width dimension.
 * @param value Padding constant.
 * @return New padded tensor on success, or NULL on error.
 */
FloatTensor *st_pad_nchw(const FloatTensor *input, size_t pad_h, size_t pad_w,
                         float value);
/* ---- MPSGraph warmup (shape-aware, reduces first-run latency) ---- */

/**
 * @brief One warmup shape descriptor for MPS graph pre-compilation.
 *
 * Set `c_out = 0` to skip Conv2D warmup for the entry.
 */
typedef struct {
  size_t n, c_in, h, w;    /* input: batch, channels, height, width */
  size_t c_out;             /* conv: output channels (0 = conv skipped) */
  size_t kh, kw;            /* conv/pool: kernel size                   */
  size_t sh, sw;            /* stride                                   */
  size_t ph, pw;            /* padding                                  */
} StWarmupShape;

/**
 * @brief Pre-compile MPS graphs for a list of tensor shapes.
 * @param shapes Array of warmup descriptors.
 * @param count Number of entries in @p shapes.
 *
 * Warmed paths include Conv2D (when `c_out > 0`), MaxPool2D, AvgPool2D,
 * and BatchNorm2D.
 */
void st_mps_warmup_shapes(const StWarmupShape *shapes, size_t count);
#endif  // ST_H
