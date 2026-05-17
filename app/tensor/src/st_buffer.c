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

static _Atomic uint64_t g_pending_samples = 0u;
static _Atomic uint64_t g_pending_total_depth = 0u;
static _Atomic uint64_t g_pending_enqueued = 0u;
static _Atomic uint64_t g_pending_evicted = 0u;
static _Atomic size_t g_pending_max_depth = 0u;

void st_buffer_pending_stats_reset(void) {
  atomic_store_explicit(&g_pending_samples, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_pending_total_depth, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_pending_enqueued, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_pending_evicted, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_pending_max_depth, 0u, memory_order_relaxed);
}

StBufferPendingStats st_buffer_pending_stats_get(void) {
  StBufferPendingStats s;
  s.samples = atomic_load_explicit(&g_pending_samples, memory_order_relaxed);
  s.total_depth =
      atomic_load_explicit(&g_pending_total_depth, memory_order_relaxed);
  s.enqueued = atomic_load_explicit(&g_pending_enqueued, memory_order_relaxed);
  s.evicted = atomic_load_explicit(&g_pending_evicted, memory_order_relaxed);
  s.max_depth =
      atomic_load_explicit(&g_pending_max_depth, memory_order_relaxed);
  return s;
}

void st_buffer_metal_allocator_stats_reset(void) {
#if defined(USE_ACCELERATE) && defined(__APPLE__)
  st_buffer_metal_allocator_stats_reset_impl();
#endif
}

StBufferMetalAllocatorStats st_buffer_metal_allocator_stats_get(void) {
  StBufferMetalAllocatorStats s = {0};
#if defined(USE_ACCELERATE) && defined(__APPLE__)
  s = st_buffer_metal_allocator_stats_get_impl();
#endif
  return s;
}

static void st_buffer_pending_reset(StBuffer *buf) {
  if (!buf) {
    return;
  }
  buf->_async_cmd_buf = NULL;
  buf->_async_cmd_head = 0u;
  buf->_async_cmd_count = 0u;
  for (size_t i = 0; i < ST_BUFFER_PENDING_CMDS_MAX; ++i) {
    buf->_async_cmd_ring[i] = NULL;
  }
}

static size_t st_buffer_pending_collect(StBuffer *buf, void **out_handles,
                                        size_t max_handles, bool clear_state) {
  if (!buf || !out_handles || max_handles == 0u) {
    return 0u;
  }

  size_t n = 0u;
  if (buf->_async_cmd_count > 0u) {
    size_t count = (size_t)buf->_async_cmd_count;
    for (size_t i = 0; i < count && n < max_handles; ++i) {
      const size_t idx =
          ((size_t)buf->_async_cmd_head + i) % ST_BUFFER_PENDING_CMDS_MAX;
      void *h = buf->_async_cmd_ring[idx];
      if (h) {
        out_handles[n++] = h;
      }
      if (clear_state) {
        buf->_async_cmd_ring[idx] = NULL;
      }
    }
    if (clear_state) {
      st_buffer_pending_reset(buf);
    }
    return n;
  }

  /* Legacy/manual sentinel path used by tests and guard checks. */
  if (buf->_async_cmd_buf) {
    out_handles[n++] = buf->_async_cmd_buf;
    if (clear_state) {
      buf->_async_cmd_buf = NULL;
    }
  }
  return n;
}

static size_t st_buffer_env_size_t(const char *name, size_t fallback,
                                   size_t max_value) {
  if (!name) {
    return fallback;
  }
  const char *v = getenv(name);
  if (!v || v[0] == '\0') {
    return fallback;
  }
  char *end = NULL;
  unsigned long parsed = strtoul(v, &end, 10);
  if (end == v || (end && *end != '\0')) {
    return fallback;
  }
  size_t out = (size_t)parsed;
  if (out > max_value) {
    out = max_value;
  }
  return out;
}

static bool st_buffer_env_present(const char *name) {
  if (!name) {
    return false;
  }
  const char *v = getenv(name);
  return v && v[0] != '\0';
}

static size_t st_buffer_resolve_auto_sync_every(void) {
  /* Priority: explicit numeric override > profile preset > default(0). */
  if (st_buffer_env_present("MMATRIX_ST_ASYNC_SYNC_EVERY")) {
    return st_buffer_env_size_t("MMATRIX_ST_ASYNC_SYNC_EVERY", 0u,
                                ST_BUFFER_PENDING_CMDS_MAX);
  }

  const char *profile = getenv("MMATRIX_ST_ASYNC_PROFILE");
  if (!profile || profile[0] == '\0') {
    return 0u;
  }

  if (strcmp(profile, "throughput") == 0 || strcmp(profile, "THROUGHPUT") == 0) {
    return 0u;
  }
  if (strcmp(profile, "balanced") == 0 || strcmp(profile, "BALANCED") == 0) {
    return 4u;
  }
  if (strcmp(profile, "stable") == 0 || strcmp(profile, "STABLE") == 0 ||
      strcmp(profile, "latency") == 0 || strcmp(profile, "LATENCY") == 0) {
    return 8u;
  }

  return 0u;
}

void st_buffer_track_pending_cmd(StBuffer *buf, void *cmd_handle) {
  if (!buf || !cmd_handle) {
    return;
  }

  if (buf->_async_cmd_count >= ST_BUFFER_PENDING_CMDS_MAX) {
    const size_t evict_idx = (size_t)buf->_async_cmd_head;
    void *evict = buf->_async_cmd_ring[evict_idx];
    if (evict) {
#if defined(USE_ACCELERATE) && defined(__APPLE__)
      st_buffer_metal_discard_handle(evict);
#endif
    }
    buf->_async_cmd_ring[evict_idx] = NULL;
    atomic_fetch_add_explicit(&g_pending_evicted, 1u, memory_order_relaxed);
    buf->_async_cmd_head =
        (unsigned char)(((size_t)buf->_async_cmd_head + 1u) %
                        ST_BUFFER_PENDING_CMDS_MAX);
    buf->_async_cmd_count = (unsigned char)(ST_BUFFER_PENDING_CMDS_MAX - 1u);
  }

  const size_t write_idx =
      ((size_t)buf->_async_cmd_head + (size_t)buf->_async_cmd_count) %
      ST_BUFFER_PENDING_CMDS_MAX;
  buf->_async_cmd_ring[write_idx] = cmd_handle;
  buf->_async_cmd_count = (unsigned char)((size_t)buf->_async_cmd_count + 1u);
  buf->_async_cmd_buf = cmd_handle;

  const size_t depth = (size_t)buf->_async_cmd_count;
  atomic_fetch_add_explicit(&g_pending_enqueued, 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&g_pending_samples, 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&g_pending_total_depth, (uint64_t)depth,
                            memory_order_relaxed);
  size_t cur_max =
      atomic_load_explicit(&g_pending_max_depth, memory_order_relaxed);
  while (depth > cur_max &&
         !atomic_compare_exchange_weak_explicit(
             &g_pending_max_depth, &cur_max, depth, memory_order_relaxed,
             memory_order_relaxed)) {
  }

#if defined(USE_ACCELERATE) && defined(__APPLE__)
  if (buf->type == ST_BUFFER_METAL) {
    const size_t auto_sync_every = st_buffer_resolve_auto_sync_every();
    if (auto_sync_every > 0u && depth >= auto_sync_every) {
      void *pending_handles[ST_BUFFER_PENDING_CMDS_MAX] = {0};
      const size_t pending_count =
          st_buffer_pending_collect(buf, pending_handles,
                                    ST_BUFFER_PENDING_CMDS_MAX,
                                    /*clear_state=*/true);
      for (size_t i = 0; i < pending_count; ++i) {
        st_buffer_metal_wait_handle(pending_handles[i], buf,
                                    ST_BUFFER_WAIT_REASON_BOUNDARY);
      }
    }
  }
#endif
}

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

static bool st_buffer_env_true(const char *name) {
  if (!name) {
    return false;
  }
  const char *v = getenv(name);
  if (!v || v[0] == '\0') {
    return false;
  }
  if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 ||
      strcmp(v, "TRUE") == 0 || strcmp(v, "yes") == 0 ||
      strcmp(v, "YES") == 0 || strcmp(v, "on") == 0 ||
      strcmp(v, "ON") == 0) {
    return true;
  }
  return false;
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
  st_buffer_pending_reset(buf);

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
  st_buffer_pending_reset(buf);

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
  st_buffer_pending_reset(buf);

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
  const bool force_blocking_release =
      st_buffer_env_true("MMATRIX_ST_BUFFER_RELEASE_BLOCKING");
  void *pending_handles[ST_BUFFER_PENDING_CMDS_MAX] = {0};
  const size_t pending_count =
      st_buffer_pending_collect(buf, pending_handles, ST_BUFFER_PENDING_CMDS_MAX,
                                /*clear_state=*/true);

  if (pending_count > 0u && buf->_backend_handle) {
    if (!force_blocking_release) {
      /* Non-blocking path: defer handle release until GPU work completes. */
      void *metal_handle = buf->_backend_handle;
      if (st_buffer_metal_schedule_release_many(pending_handles, pending_count,
                                                metal_handle)) {
        buf->_backend_handle = NULL;
        buf->data = NULL;
        free(buf);
        return;
      }
    }

    /* Fallback path if scheduling failed for any reason. */
    for (size_t i = 0; i < pending_count; ++i) {
      st_buffer_metal_wait_handle(pending_handles[i], buf,
                                  ST_BUFFER_WAIT_REASON_RELEASE);
    }
  } else if (pending_count > 0u) {
    /* No owned Metal storage left to protect.
     * For Metal buffers, release the retained command-buffer handle.
     * For CPU buffers (tests may inject sentinel values), just clear marker. */
    if (buf->type == ST_BUFFER_METAL) {
      for (size_t i = 0; i < pending_count; ++i) {
        st_buffer_metal_discard_handle(pending_handles[i]);
      }
    } else {
      st_buffer_pending_reset(buf);
    }
  }
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

void st_buffer_wait_gpu(StBuffer *buf) {
  if (!buf) {
    return;
  }
#if defined(USE_ACCELERATE) && defined(__APPLE__)
  void *pending_handles[ST_BUFFER_PENDING_CMDS_MAX] = {0};
  const size_t pending_count =
      st_buffer_pending_collect(buf, pending_handles, ST_BUFFER_PENDING_CMDS_MAX,
                                /*clear_state=*/true);
  for (size_t i = 0; i < pending_count; ++i) {
    st_buffer_metal_wait_handle(pending_handles[i], buf,
                                ST_BUFFER_WAIT_REASON_BOUNDARY);
  }
#else
  (void)buf;
#endif
}

bool st_buffer_last_gpu_elapsed_ms(const StBuffer *buf, double *out_ms) {
  if (!buf || !out_ms || !buf->_last_gpu_elapsed_valid) {
    return false;
  }
  *out_ms = buf->_last_gpu_elapsed_ms;
  return true;
}

bool st_buffer_last_gpu_profile(const StBuffer *buf,
                                StBufferGpuProfile *out_profile) {
  if (!buf || !out_profile || !buf->_last_gpu_profile_valid) {
    return false;
  }
  *out_profile = buf->_last_gpu_profile;
  return true;
}
