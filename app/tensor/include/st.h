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

#include "st_buffer.h"
#include "st_dtype.h"

#define ST_MAX_DIMS 8

typedef enum StLayout {
  ST_LAYOUT_CONTIGUOUS = 0,
  ST_LAYOUT_NCHW = 1,
  ST_LAYOUT_NHWC = 2,
  ST_LAYOUT_TBF = 3,  // [time, batch, feature]
  ST_LAYOUT_BTF = 4,  // [batch, time, feature]
} StLayout;

typedef struct FloatMatrix FloatMatrix;

typedef struct FloatTensor {
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
  void (*extra_free)(void *);        // destructor for extra (NULL = no-op)
} FloatTensor;

// Compute default row-major contiguous strides from shape.
bool st_compute_default_strides(size_t ndim, const size_t *shape,
                                ptrdiff_t *out_strides);

// Compute number of elements from shape.
bool st_numel_from_shape(size_t ndim, const size_t *shape, size_t *out_numel);

// Create contiguous tensor and zero-initialize memory.
FloatTensor *st_create(size_t ndim, const size_t *shape);

// Create contiguous bf16 tensor and zero-initialize memory (half the storage of f32).
FloatTensor *st_create_bf16(size_t ndim, const size_t *shape);

// Create tensor wrapper around existing data pointer (optional ownership transfer).
FloatTensor *st_create_with_data(size_t ndim, const size_t *shape, float *data,
                                 size_t capacity, bool take_ownership);

// Create deep copy of tensor in contiguous storage (preserves dtype).
FloatTensor *st_clone(const FloatTensor *src);

// Convert tensor to f32 (returns new tensor; no-op clone if already f32).
FloatTensor *st_to_f32(const FloatTensor *src);

// Convert tensor to bf16 (returns new tensor; no-op clone if already bf16).
FloatTensor *st_to_bf16(const FloatTensor *src);

// Create tensor view with custom shape/strides and offset (view does not own data).
FloatTensor *st_view(FloatTensor *base, size_t ndim, const size_t *shape,
                     const ptrdiff_t *strides, size_t offset_elements);

// Return true when tensor uses default row-major contiguous layout.
bool st_is_contiguous(const FloatTensor *tensor);

// Reshape tensor in-place (requires contiguous layout and unchanged numel).
bool st_reshape(FloatTensor *tensor, size_t new_ndim, const size_t *new_shape);

// Create permuted tensor view (no data copy).
FloatTensor *st_permute_view(FloatTensor *base, const size_t *perm);

// Read scalar value at multi-index.
float st_get(const FloatTensor *tensor, const size_t *indices);

// Write scalar value at multi-index.
bool st_set(FloatTensor *tensor, const size_t *indices, float value);

// Expose 2D contiguous tensor as FloatMatrix view (no copy, no data ownership transfer).
bool st_as_sm_view(const FloatTensor *tensor, FloatMatrix *out_view);

// Destroy tensor and owned data when owns_data == true.
void st_destroy(FloatTensor *tensor);

// ---- Element-wise in-place operations ----

// a[i] += b[i] for all elements (broadcast: b may be NULL → no-op).
bool st_inplace_add(FloatTensor *a, const FloatTensor *b);

// a[i] -= b[i] for all elements.
bool st_inplace_sub(FloatTensor *a, const FloatTensor *b);

// t[i] *= scalar for all elements.
bool st_inplace_scale(FloatTensor *t, float scalar);

// a[i] *= b[i] for all elements (Hadamard product).
bool st_inplace_elementwise_multiply(FloatTensor *a, const FloatTensor *b);

// Fill all elements with given value.
bool st_fill(FloatTensor *t, float value);

// ---- Activation functions on tensors ----

// ReLU: t[i] = max(0, t[i]) in-place.
bool st_apply_relu(FloatTensor *t);

// ReLU backward: grad[i] = (activation[i] > 0) ? grad[i] : 0  in-place.
bool st_apply_relu_backward(const FloatTensor *activation, FloatTensor *grad);

// ---- Reduction ----

// Sum over specified axes, collapsing them.
// Returns a new tensor with the summed axes removed.
// axes: array of axis indices to sum over (e.g. {0,2,3} for bias-gradient NCHW).
FloatTensor *st_sum_axes(const FloatTensor *t, const size_t *axes,
                         size_t num_axes);

// ---- Padding ----

// Zero-pad (or constant-pad) an NCHW tensor along H and W.
// Returns a new tensor with shape [N, C, H+2*pad_h, W+2*pad_w].
FloatTensor *st_pad_nchw(const FloatTensor *input, size_t pad_h, size_t pad_w,
                         float value);
/* ---- MPSGraph warmup (shape-aware, reduces first-run latency) ---- */

/// Shape descriptor for st_mps_warmup_shapes().
/// Set c_out=0 to skip Conv2D warmup for this entry.
typedef struct {
  size_t n, c_in, h, w;    /* input: batch, channels, height, width */
  size_t c_out;             /* conv: output channels (0 = conv skipped) */
  size_t kh, kw;            /* conv/pool: kernel size                   */
  size_t sh, sw;            /* stride                                   */
  size_t ph, pw;            /* padding                                  */
} StWarmupShape;

/// Pre-compiles MPSGraph for each shape in `shapes[0..count-1]`.
/// Warms up Conv2D (if c_out>0), MaxPool2D, AvgPool2D and BatchNorm2D.
/// Safe to call from any thread; silently skips shapes where MPS is not
/// the selected backend or where any allocation fails.
/// Activated automatically when MMATRIX_ST_MPS_WARMUP=1 is set; callers
/// can also invoke this function directly with custom shape lists.
void st_mps_warmup_shapes(const StWarmupShape *shapes, size_t count);
#endif  // ST_H
