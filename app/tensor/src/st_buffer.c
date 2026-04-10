/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_buffer.c — StBuffer implementation (CPU + Metal dispatch).
 */

#include "st_buffer.h"

#include <log.h>
#include <stdlib.h>
#include <string.h>

#if defined(USE_ACCELERATE) && defined(__APPLE__)
#include "st_buffer_metal.h"
#endif

/* ------------------------------------------------------------------ */
/*  CPU allocation                                                     */
/* ------------------------------------------------------------------ */

StBuffer *st_buffer_alloc_cpu(size_t num_floats) {
  if (num_floats == 0) {
    log_error("Error: st_buffer_alloc_cpu zero size.");
    return NULL;
  }

  StBuffer *buf = (StBuffer *)calloc(1, sizeof(StBuffer));
  if (!buf) {
    log_error("Error: st_buffer_alloc_cpu struct allocation failed.");
    return NULL;
  }

  buf->data = (float *)calloc(num_floats, sizeof(float));
  if (!buf->data) {
    log_error("Error: st_buffer_alloc_cpu data allocation failed.");
    free(buf);
    return NULL;
  }

  buf->type = ST_BUFFER_CPU;
  buf->size_bytes = num_floats * sizeof(float);
  buf->capacity = num_floats;
  buf->refcount = 1;
  buf->owns_data = true;
  buf->_backend_handle = NULL;

  return buf;
}

/* ------------------------------------------------------------------ */
/*  Metal allocation                                                   */
/* ------------------------------------------------------------------ */

StBuffer *st_buffer_alloc_metal(size_t num_floats) {
#if defined(USE_ACCELERATE) && defined(__APPLE__)
  StBuffer *buf = st_buffer_alloc_metal_impl(num_floats);
  if (buf) {
    return buf;
  }
  /* Fallback to CPU if Metal allocation fails. */
#endif
  return st_buffer_alloc_cpu(num_floats);
}

/* ------------------------------------------------------------------ */
/*  Platform-best allocation                                           */
/* ------------------------------------------------------------------ */

StBuffer *st_buffer_alloc(size_t num_floats) {
#if defined(USE_ACCELERATE) && defined(__APPLE__)
  return st_buffer_alloc_metal(num_floats);
#else
  return st_buffer_alloc_cpu(num_floats);
#endif
}

/* ------------------------------------------------------------------ */
/*  Wrap existing pointer                                              */
/* ------------------------------------------------------------------ */

StBuffer *st_buffer_from_ptr(float *data, size_t num_floats,
                             bool take_ownership) {
  if (data == NULL || num_floats == 0) {
    log_error("Error: st_buffer_from_ptr invalid input.");
    return NULL;
  }

  StBuffer *buf = (StBuffer *)calloc(1, sizeof(StBuffer));
  if (!buf) {
    log_error("Error: st_buffer_from_ptr struct allocation failed.");
    return NULL;
  }

  buf->type = ST_BUFFER_CPU;
  buf->data = data;
  buf->size_bytes = num_floats * sizeof(float);
  buf->capacity = num_floats;
  buf->refcount = 1;
  buf->owns_data = take_ownership;
  buf->_backend_handle = NULL;

  return buf;
}

/* ------------------------------------------------------------------ */
/*  Reference counting                                                 */
/* ------------------------------------------------------------------ */

StBuffer *st_buffer_retain(StBuffer *buf) {
  if (buf) {
    buf->refcount++;
  }
  return buf;
}

void st_buffer_release(StBuffer *buf) {
  if (!buf) {
    return;
  }

  buf->refcount--;
  if (buf->refcount > 0) {
    return;
  }

  /* Refcount reached 0 — free backing storage. */
#if defined(USE_ACCELERATE) && defined(__APPLE__)
  if (buf->_backend_handle) {
    /* Metal buffer: release the MTLBuffer; data pointer is owned by it. */
    st_buffer_release_metal_handle(buf->_backend_handle);
  } else
#endif
  {
    if (buf->owns_data && buf->data) {
      free(buf->data);
    }
  }

  buf->data = NULL;
  buf->_backend_handle = NULL;
  free(buf);
}
