/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_buffer.c — StBuffer implementation (CPU + Metal dispatch).
 */

#include "st_buffer.h"

#include <log.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(USE_ACCELERATE) && defined(__APPLE__)
#include "st_buffer_metal.h"
#endif

/* ------------------------------------------------------------------ */
/*  CPU allocation                                                     */
/* ------------------------------------------------------------------ */

static bool st_buffer_num_floats_to_bytes(size_t num_floats,
                                          size_t *out_bytes) {
  if (out_bytes == NULL || num_floats == 0 ||
      num_floats > SIZE_MAX / sizeof(float)) {
    return false;
  }

  *out_bytes = num_floats * sizeof(float);
  return true;
}

static bool st_buffer_align_bytes(size_t num_bytes, size_t *out_aligned_bytes) {
  if (out_aligned_bytes == NULL || num_bytes == 0 || num_bytes > SIZE_MAX - 63u) {
    return false;
  }

  *out_aligned_bytes = (num_bytes + 63u) & ~(size_t)63u;
  return true;
}

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

  size_t alloc_bytes = 0;
  if (!st_buffer_num_floats_to_bytes(num_floats, &alloc_bytes)) {
    log_error("Error: st_buffer_alloc_cpu size overflow.");
    free(buf);
    return NULL;
  }

  void *raw = NULL;
  if (posix_memalign(&raw, 64, alloc_bytes) != 0 || !raw) {
    log_error("Error: st_buffer_alloc_cpu data allocation failed.");
    free(buf);
    return NULL;
  }
  memset(raw, 0, alloc_bytes);
  buf->data = (float *)raw;

  buf->type = ST_BUFFER_CPU;
  buf->size_bytes = alloc_bytes;
  buf->capacity = num_floats;
  buf->refcount = 1;
  buf->owns_data = true;
  buf->_backend_handle = NULL;

  return buf;
}

/* ------------------------------------------------------------------ */
/*  Metal allocation                                                   */
/* ------------------------------------------------------------------ */

StBuffer *st_buffer_alloc_bytes_cpu(size_t num_bytes) {
  if (num_bytes == 0) {
    log_error("Error: st_buffer_alloc_bytes_cpu zero size.");
    return NULL;
  }

  StBuffer *buf = (StBuffer *)calloc(1, sizeof(StBuffer));
  if (!buf) {
    log_error("Error: st_buffer_alloc_bytes_cpu struct allocation failed.");
    return NULL;
  }

  size_t alloc_bytes = 0;
  if (!st_buffer_align_bytes(num_bytes, &alloc_bytes)) {
    log_error("Error: st_buffer_alloc_bytes_cpu size overflow.");
    free(buf);
    return NULL;
  }

  void *raw = NULL;
  if (posix_memalign(&raw, 64, alloc_bytes) != 0 || !raw) {
    log_error("Error: st_buffer_alloc_bytes_cpu data allocation failed.");
    free(buf);
    return NULL;
  }
  memset(raw, 0, alloc_bytes);
  buf->data = (float *)raw;

  buf->type = ST_BUFFER_CPU;
  buf->size_bytes = num_bytes;
  buf->capacity = num_bytes / sizeof(float);
  buf->refcount = 1;
  buf->owns_data = true;
  buf->_backend_handle = NULL;

  return buf;
}

/* ------------------------------------------------------------------ */
/*  Metal allocation (float-based)                                     */
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

StBuffer *st_buffer_alloc_bytes(size_t num_bytes) {
  /* For now always CPU — Metal byte-based alloc can be added later. */
  return st_buffer_alloc_bytes_cpu(num_bytes);
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

  size_t alloc_bytes = 0;
  if (!st_buffer_num_floats_to_bytes(num_floats, &alloc_bytes)) {
    log_error("Error: st_buffer_from_ptr size overflow.");
    free(buf);
    return NULL;
  }

  buf->type = ST_BUFFER_CPU;
  buf->data = data;
  buf->size_bytes = alloc_bytes;
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
    atomic_fetch_add(&buf->refcount, 1);
  }
  return buf;
}

void st_buffer_release(StBuffer *buf) {
  if (!buf) {
    return;
  }

  if (atomic_fetch_sub(&buf->refcount, 1) > 1) {
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
