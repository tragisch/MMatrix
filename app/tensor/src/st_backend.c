/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "st_backend.h"

#include <log.h>
#include <stdatomic.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/*  Global default backend                                             */
/* ------------------------------------------------------------------ */

static const StBackend *g_default_backend = NULL;  /* NULL = auto-dispatch */

void st_set_default_backend(const StBackend *backend) {
  g_default_backend = backend;
}

const StBackend *st_get_default_backend(void) { return g_default_backend; }

/* ------------------------------------------------------------------ */
/*  Fallback reason (thread-local)                                     */
/* ------------------------------------------------------------------ */

static _Thread_local StBackendFallbackReason g_last_fallback_reason =
    ST_FALLBACK_NONE;

void st_set_last_fallback_reason(StBackendFallbackReason reason) {
  g_last_fallback_reason = reason;
}

StBackendFallbackReason st_get_last_fallback_reason(void) {
  return g_last_fallback_reason;
}

const char *st_fallback_reason_str(StBackendFallbackReason reason) {
  switch (reason) {
    case ST_FALLBACK_NONE:              return "none";
    case ST_FALLBACK_NO_MPS_DEVICE:     return "no_mps_device";
    case ST_FALLBACK_BACKEND_FORCED_CPU:return "backend_forced_cpu";
    case ST_FALLBACK_THRESHOLD:         return "threshold";
    case ST_FALLBACK_MPS_ERROR:         return "mps_error";
    case ST_FALLBACK_DTYPE_UNSUPPORTED: return "dtype_unsupported";
    default:                            return "unknown";
  }
}

/* ------------------------------------------------------------------ */
/*  Backend dispatch counters                                          */
/* ------------------------------------------------------------------ */

static atomic_long g_counter_mps_hit      = 0;
static atomic_long g_counter_mps_miss     = 0;
static atomic_long g_counter_fallback_gemm = 0;
static atomic_long g_counter_fallback_ref  = 0;
static atomic_long g_counter_conv_readbytes = 0;
static atomic_long g_counter_conv_fastpath_hit = 0;
static atomic_long g_counter_conv_fastpath_executable_nil = 0;
static atomic_long g_counter_conv_fastpath_missing_feed = 0;
static atomic_long g_counter_conv_fastpath_preout_nil = 0;
static atomic_long g_counter_conv_fastpath_cmd_buf_nil = 0;
static atomic_long g_counter_conv_fastpath_encode_exception = 0;

void st_backend_counter_mps_hit(void) {
  atomic_fetch_add_explicit(&g_counter_mps_hit, 1L, memory_order_relaxed);
}

void st_backend_counter_mps_miss(void) {
  atomic_fetch_add_explicit(&g_counter_mps_miss, 1L, memory_order_relaxed);
}

void st_backend_counter_fallback_gemm(void) {
  atomic_fetch_add_explicit(&g_counter_fallback_gemm, 1L, memory_order_relaxed);
}

void st_backend_counter_fallback_ref(void) {
  atomic_fetch_add_explicit(&g_counter_fallback_ref, 1L, memory_order_relaxed);
}

void st_backend_counter_conv_readbytes(void) {
  atomic_fetch_add_explicit(&g_counter_conv_readbytes, 1L, memory_order_relaxed);
}

void st_backend_counter_conv_fastpath_hit(void) {
  atomic_fetch_add_explicit(&g_counter_conv_fastpath_hit, 1L, memory_order_relaxed);
}

void st_backend_counter_conv_fastpath_executable_nil(void) {
  atomic_fetch_add_explicit(&g_counter_conv_fastpath_executable_nil, 1L,
                            memory_order_relaxed);
}

void st_backend_counter_conv_fastpath_missing_feed(void) {
  atomic_fetch_add_explicit(&g_counter_conv_fastpath_missing_feed, 1L,
                            memory_order_relaxed);
}

void st_backend_counter_conv_fastpath_preout_nil(void) {
  atomic_fetch_add_explicit(&g_counter_conv_fastpath_preout_nil, 1L,
                            memory_order_relaxed);
}

void st_backend_counter_conv_fastpath_cmd_buf_nil(void) {
  atomic_fetch_add_explicit(&g_counter_conv_fastpath_cmd_buf_nil, 1L,
                            memory_order_relaxed);
}

void st_backend_counter_conv_fastpath_encode_exception(void) {
  atomic_fetch_add_explicit(&g_counter_conv_fastpath_encode_exception, 1L,
                            memory_order_relaxed);
}

StBackendCounters st_backend_get_counters(void) {
  StBackendCounters c;
  c.mps_hit       = atomic_load_explicit(&g_counter_mps_hit,       memory_order_relaxed);
  c.mps_miss      = atomic_load_explicit(&g_counter_mps_miss,      memory_order_relaxed);
  c.fallback_gemm = atomic_load_explicit(&g_counter_fallback_gemm, memory_order_relaxed);
  c.fallback_ref  = atomic_load_explicit(&g_counter_fallback_ref,  memory_order_relaxed);
  c.conv_readbytes = atomic_load_explicit(&g_counter_conv_readbytes, memory_order_relaxed);
  c.conv_fastpath_hit = atomic_load_explicit(&g_counter_conv_fastpath_hit, memory_order_relaxed);
    c.conv_fastpath_executable_nil =
      atomic_load_explicit(&g_counter_conv_fastpath_executable_nil,
                 memory_order_relaxed);
    c.conv_fastpath_missing_feed =
      atomic_load_explicit(&g_counter_conv_fastpath_missing_feed,
                 memory_order_relaxed);
    c.conv_fastpath_preout_nil =
      atomic_load_explicit(&g_counter_conv_fastpath_preout_nil,
                 memory_order_relaxed);
    c.conv_fastpath_cmd_buf_nil =
      atomic_load_explicit(&g_counter_conv_fastpath_cmd_buf_nil,
                 memory_order_relaxed);
    c.conv_fastpath_encode_exception =
      atomic_load_explicit(&g_counter_conv_fastpath_encode_exception,
                 memory_order_relaxed);
  return c;
}

void st_backend_reset_counters(void) {
  atomic_store_explicit(&g_counter_mps_hit,       0L, memory_order_relaxed);
  atomic_store_explicit(&g_counter_mps_miss,      0L, memory_order_relaxed);
  atomic_store_explicit(&g_counter_fallback_gemm, 0L, memory_order_relaxed);
  atomic_store_explicit(&g_counter_fallback_ref,  0L, memory_order_relaxed);
  atomic_store_explicit(&g_counter_conv_readbytes, 0L, memory_order_relaxed);
  atomic_store_explicit(&g_counter_conv_fastpath_hit, 0L, memory_order_relaxed);
  atomic_store_explicit(&g_counter_conv_fastpath_executable_nil, 0L,
                        memory_order_relaxed);
  atomic_store_explicit(&g_counter_conv_fastpath_missing_feed, 0L,
                        memory_order_relaxed);
  atomic_store_explicit(&g_counter_conv_fastpath_preout_nil, 0L,
                        memory_order_relaxed);
  atomic_store_explicit(&g_counter_conv_fastpath_cmd_buf_nil, 0L,
                        memory_order_relaxed);
  atomic_store_explicit(&g_counter_conv_fastpath_encode_exception, 0L,
                        memory_order_relaxed);
}

/* ------------------------------------------------------------------ */
/*  Auto-select                                                        */
/* ------------------------------------------------------------------ */

const StBackend *st_select_backend(StOp op, const FloatTensor *tensor) {
  /* Honour explicit override. */
  if (g_default_backend) {
    if (g_default_backend->supports_op &&
        g_default_backend->supports_op(
            op, tensor ? tensor->numel : 0,
            (tensor && tensor->buf) ? tensor->buf->type : ST_BUFFER_CPU)) {
      g_last_fallback_reason = ST_FALLBACK_NONE;
      return g_default_backend;
    }
    /* Explicit backend doesn't support this op → fall through to CPU. */
    g_last_fallback_reason = ST_FALLBACK_BACKEND_FORCED_CPU;
    log_debug("backend: op=%d forced backend '%s' does not support op → CPU "
              "(reason=%s)", op,
              g_default_backend->name ? g_default_backend->name : "?",
              st_fallback_reason_str(ST_FALLBACK_BACKEND_FORCED_CPU));
    return NULL;
  }

  /* Auto mode: try MPS first. */
  const StBackend *mps = st_backend_mps();
  if (!mps) {
    g_last_fallback_reason = ST_FALLBACK_NO_MPS_DEVICE;
    return NULL;
  }

  if (mps->supports_op) {
    StBufferType bt =
        (tensor && tensor->buf) ? tensor->buf->type : ST_BUFFER_CPU;
    if (mps->supports_op(op, tensor ? tensor->numel : 0, bt)) {
      g_last_fallback_reason = ST_FALLBACK_NONE;
      atomic_fetch_add_explicit(&g_counter_mps_hit, 1L, memory_order_relaxed);
      return mps;
    }
  }

  /* MPS available but declined this op (threshold / dtype). */
  g_last_fallback_reason = ST_FALLBACK_THRESHOLD;
  atomic_fetch_add_explicit(&g_counter_mps_miss, 1L, memory_order_relaxed);
  log_debug("backend: op=%d MPS declined → CPU (reason=%s)", op,
            st_fallback_reason_str(ST_FALLBACK_THRESHOLD));

  /* CPU fallback — caller uses existing implementation directly. */
  return NULL;
}

/* ------------------------------------------------------------------ */
/*  Non-Apple stub for st_backend_mps()                                */
/* ------------------------------------------------------------------ */

#if !(defined(__APPLE__) && defined(USE_ACCELERATE))
const StBackend *st_backend_mps(void) { return NULL; }
bool st_backend_conv2d_batchnorm2d_forward_mps(
    const FloatTensor *input, const FloatTensor *weight,
    const FloatTensor *bias, const StConv2dParams *params,
    const FloatTensor *gamma, const FloatTensor *beta, float epsilon,
    FloatTensor *output, FloatTensor *mean, FloatTensor *var,
    bool apply_relu) {
  (void)input;
  (void)weight;
  (void)bias;
  (void)params;
  (void)gamma;
  (void)beta;
  (void)epsilon;
  (void)output;
  (void)mean;
  (void)var;
  (void)apply_relu;
  return false;
}

bool st_backend_conv2d_batchnorm2d_pool_forward_mps(
    const FloatTensor *input, const FloatTensor *weight,
    const FloatTensor *bias, const StConv2dParams *conv_params,
    const FloatTensor *gamma, const FloatTensor *beta, float epsilon,
    const StPool2dParams *pool_params,
    FloatTensor *output, FloatTensor *mean, FloatTensor *var,
    bool apply_relu) {
  (void)input;
  (void)weight;
  (void)bias;
  (void)conv_params;
  (void)gamma;
  (void)beta;
  (void)epsilon;
  (void)pool_params;
  (void)output;
  (void)mean;
  (void)var;
  (void)apply_relu;
  return false;
}

bool st_backend_set_mps_thresholds(size_t pool_threshold,
                                   size_t batchnorm_threshold) {
  (void)pool_threshold;
  (void)batchnorm_threshold;
  return false;
}

void st_backend_get_mps_thresholds(size_t *out_pool_threshold,
                                   size_t *out_batchnorm_threshold) {
  if (out_pool_threshold != NULL) {
    *out_pool_threshold = 0u;
  }
  if (out_batchnorm_threshold != NULL) {
    *out_batchnorm_threshold = 0u;
  }
}

void st_backend_reload_mps_thresholds_from_env(void) {}
#endif
