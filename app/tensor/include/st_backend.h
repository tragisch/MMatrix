/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_backend.h — Backend dispatch interface for tensor operations
 *
 * Each backend (CPU, MPS) registers a vtable of function pointers.
 * The dispatch layer selects the best backend at runtime based on
 * tensor size, buffer type and availability.
 *
 * Inspired by ggml's ggml_backend_i / ggml_backend_device_i pattern
 * but kept minimal: no dynamic loading, no registry — just a struct
 * of function pointers with a single global default + optional override.
 */

#ifndef ST_BACKEND_H
#define ST_BACKEND_H

#include <stdbool.h>
#include <stddef.h>

#include "st.h"
#include "st_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Op enum — every dispatchable operation                             */
/* ------------------------------------------------------------------ */

typedef enum StOp {
  /* conv */
  ST_OP_CONV2D_FORWARD,

  /* pool */
  ST_OP_MAXPOOL2D_FORWARD,
  ST_OP_AVGPOOL2D_FORWARD,

  /* normalization */
  ST_OP_BATCHNORM2D_FORWARD,

  ST_OP_COUNT  /* sentinel */
} StOp;

/* ------------------------------------------------------------------ */
/*  Forward declarations for param structs                             */
/* ------------------------------------------------------------------ */

typedef struct StConv2dParams StConv2dParams;  /* defined in st_conv.h */

/* ------------------------------------------------------------------ */
/*  Backend vtable                                                     */
/* ------------------------------------------------------------------ */

typedef struct StBackend {
  /* Human-readable name: "cpu", "mps", … */
  const char *name;

  /* ---- Capabilities ---- */

  /// Return true if this backend can handle `op` for a tensor with
  /// `numel` elements stored in `buf_type`.  Used by the auto-dispatch
  /// layer to choose the best backend.
  bool (*supports_op)(StOp op, size_t numel, StBufferType buf_type);

  /* ---- Conv2D ---- */

  /// Returns true on success (output filled), false to signal
  /// "not handled — caller should fall back to CPU".
  bool (*conv2d_forward)(const FloatTensor *input, const FloatTensor *weight,
                         const FloatTensor *bias, const StConv2dParams *params,
                         FloatTensor *output);

  /* ---- Pool ---- */

  bool (*maxpool2d_forward)(const FloatTensor *input, size_t kh, size_t kw,
                            size_t sh, size_t sw, size_t ph, size_t pw,
                            FloatTensor *output, FloatTensor *indices);

  bool (*avgpool2d_forward)(const FloatTensor *input, size_t kh, size_t kw,
                            size_t sh, size_t sw, size_t ph, size_t pw,
                            FloatTensor *output);

  /* ---- BatchNorm ---- */

  bool (*batchnorm2d_forward)(const FloatTensor *input,
                              const FloatTensor *gamma,
                              const FloatTensor *beta, float epsilon,
                              FloatTensor *output, FloatTensor *mean,
                              FloatTensor *var);

} StBackend;

/* ------------------------------------------------------------------ */
/*  Global dispatch                                                    */
/* ------------------------------------------------------------------ */

/// Return the MPS backend, or NULL if not compiled / not available.
const StBackend *st_backend_mps(void);

/// Set the default backend used by tensor dispatch.
/// Pass NULL to reset to auto-dispatch (tries MPS, falls back to CPU).
void st_set_default_backend(const StBackend *backend);

/// Get the currently active default backend, or NULL for auto-dispatch.
const StBackend *st_get_default_backend(void);

/// Auto-select the best backend for a given op and tensor.
/// Returns the MPS backend if available and `supports_op()` returns true
/// for the tensor's numel and buffer type; otherwise returns NULL
/// (meaning: use the existing CPU code path).
const StBackend *st_select_backend(StOp op, const FloatTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif /* ST_BACKEND_H */
