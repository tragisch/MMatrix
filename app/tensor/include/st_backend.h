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
typedef struct StPool2dParams StPool2dParams;  /* defined in st_conv.h */

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
/*  Fallback reason codes                                              */
/* ------------------------------------------------------------------ */

/// Reason why a backend selection fell back to CPU.
/// Updated by st_select_backend() and by individual dispatch sites.
/// Per-thread: each thread maintains its own last reason.
typedef enum StBackendFallbackReason {
  ST_FALLBACK_NONE = 0,           /**< No fallback — preferred backend used. */
  ST_FALLBACK_NO_MPS_DEVICE,      /**< MPS not compiled or no Metal device.  */
  ST_FALLBACK_BACKEND_FORCED_CPU, /**< User-set override does not support op. */
  ST_FALLBACK_THRESHOLD,          /**< Below MACs / element-count threshold.  */
  ST_FALLBACK_MPS_ERROR,          /**< MPS backend returned false at runtime. */
  ST_FALLBACK_DTYPE_UNSUPPORTED,  /**< dtype not supported by backend.        */
} StBackendFallbackReason;

/// Set the thread-local last fallback reason.
void st_set_last_fallback_reason(StBackendFallbackReason reason);

/// Get the thread-local last fallback reason.
StBackendFallbackReason st_get_last_fallback_reason(void);

/// Human-readable name for a fallback reason (never NULL).
const char *st_fallback_reason_str(StBackendFallbackReason reason);

/* ------------------------------------------------------------------ */
/*  Backend dispatch counters                                          */
/* ------------------------------------------------------------------ */

/// Counters for backend dispatch decisions.  All fields are process-wide
/// atomics (stdatomic) so they are safe to read/write from any thread.
/// Overhead in hot paths is a single atomic_fetch_add (≈1 cycle on TSO).
typedef struct StBackendCounters {
  long mps_hit;         /**< Op successfully handled by MPS.               */
  long mps_miss;        /**< MPS declined or unavailable, fell to CPU.      */
  long fallback_gemm;   /**< CPU fallback routed to GEMM path.              */
  long fallback_ref;    /**< CPU fallback routed to reference path.         */
} StBackendCounters;

/// Increment the mps_hit counter by 1.
void st_backend_counter_mps_hit(void);

/// Increment the mps_miss counter by 1.
void st_backend_counter_mps_miss(void);

/// Increment the fallback_gemm counter by 1.
void st_backend_counter_fallback_gemm(void);

/// Increment the fallback_ref counter by 1.
void st_backend_counter_fallback_ref(void);

/// Snapshot of all counters at the time of the call (non-atomic snapshot).
StBackendCounters st_backend_get_counters(void);

/// Reset all counters to zero.
void st_backend_reset_counters(void);

/* ------------------------------------------------------------------ */
/*  Global dispatch                                                    */
/* ------------------------------------------------------------------ */

/// Return the MPS backend, or NULL if not compiled / not available.
const StBackend *st_backend_mps(void);

/// MPS-specific fused Conv2D + BatchNorm2D forward helper.
/// Returns true when MPS handled the full fused path, false when caller
/// should fall back to the regular sequential implementation.
/// mean and var may be NULL (inference mode: GPU result not read back).
/// apply_relu: appends a ReLU op to the MPSGraph subgraph when true.
bool st_backend_conv2d_batchnorm2d_forward_mps(
    const FloatTensor *input, const FloatTensor *weight,
    const FloatTensor *bias, const StConv2dParams *params,
    const FloatTensor *gamma, const FloatTensor *beta, float epsilon,
    FloatTensor *output, FloatTensor *mean, FloatTensor *var,
    bool apply_relu);

/// MPS-specific fused Conv2D + BatchNorm2D + Pool2D forward helper.
/// pool_params->pool_type selects MaxPool (ST_POOL_MAX) or AvgPool (ST_POOL_AVG).
/// mean and var may be NULL (inference mode). apply_relu: ReLU after BN, before Pool.
bool st_backend_conv2d_batchnorm2d_pool_forward_mps(
    const FloatTensor *input, const FloatTensor *weight,
    const FloatTensor *bias, const StConv2dParams *conv_params,
    const FloatTensor *gamma, const FloatTensor *beta, float epsilon,
    const StPool2dParams *pool_params,
    FloatTensor *output, FloatTensor *mean, FloatTensor *var,
    bool apply_relu);

/// Override MPS AUTO dispatch thresholds for pool and batchnorm.
/// Returns false on invalid input.
bool st_backend_set_mps_thresholds(size_t pool_threshold,
                                   size_t batchnorm_threshold);

/// Query currently active MPS AUTO dispatch thresholds for pool and batchnorm.
void st_backend_get_mps_thresholds(size_t *out_pool_threshold,
                                   size_t *out_batchnorm_threshold);

/// Reload MPS AUTO dispatch thresholds for pool and batchnorm from environment:
/// `MMATRIX_ST_POOL_MPS_THRESHOLD`
/// `MMATRIX_ST_BN_MPS_THRESHOLD`
void st_backend_reload_mps_thresholds_from_env(void);

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

/// Pre-compile the MPS conv2D graph for the given shape.
/// Called by st_mps_warmup_shapes(); no-op if MPS unavailable.
void st_backend_mps_warmup_conv2d(size_t n, size_t c_in, size_t h, size_t w,
                                   size_t c_out, size_t kh, size_t kw,
                                   size_t sh, size_t sw, size_t ph, size_t pw,
                                   size_t dh, size_t dw);

#ifdef __cplusplus
}
#endif

#endif /* ST_BACKEND_H */
