/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "st_backend.h"

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
/*  Auto-select                                                        */
/* ------------------------------------------------------------------ */

const StBackend *st_select_backend(StOp op, const FloatTensor *tensor) {
  /* Honour explicit override. */
  if (g_default_backend) {
    if (g_default_backend->supports_op &&
        g_default_backend->supports_op(
            op, tensor ? tensor->numel : 0,
            (tensor && tensor->buf) ? tensor->buf->type : ST_BUFFER_CPU)) {
      return g_default_backend;
    }
    /* Explicit backend doesn't support this op → fall through to CPU. */
    return NULL;
  }

  /* Auto mode: try MPS first. */
  const StBackend *mps = st_backend_mps();
  if (mps && mps->supports_op) {
    StBufferType bt =
        (tensor && tensor->buf) ? tensor->buf->type : ST_BUFFER_CPU;
    if (mps->supports_op(op, tensor ? tensor->numel : 0, bt)) {
      return mps;
    }
  }

  /* CPU fallback — caller uses existing implementation directly. */
  return NULL;
}

/* ------------------------------------------------------------------ */
/*  Non-Apple stub for st_backend_mps()                                */
/* ------------------------------------------------------------------ */

#if !(defined(__APPLE__) && defined(USE_ACCELERATE))
const StBackend *st_backend_mps(void) { return NULL; }
#endif
