/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_buffer.h — Tensor storage abstraction
 *
 * Separates memory ownership from tensor metadata.  On Apple Silicon the
 * buffer can wrap an MTLBuffer (StorageModeShared) so that CPU and GPU
 * share the same physical memory — no copy-in / copy-out.
 *
 * On other platforms (or for plain CPU tensors) the buffer simply wraps
 * a malloc'd / aligned allocation.
 */

#ifndef ST_BUFFER_H
#define ST_BUFFER_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Buffer type tag                                                    */
/* ------------------------------------------------------------------ */

typedef enum StBufferType {
  ST_BUFFER_CPU = 0,    // plain aligned allocation (calloc / posix_memalign)
  ST_BUFFER_METAL = 1,  // MTLBuffer with StorageModeShared (Apple only)
} StBufferType;

typedef struct StBufferGpuProfile {
  double feed_ms;
  double command_ms;
  double encode_ms;
  double commit_ms;
  double sync_wait_ms;
} StBufferGpuProfile;

/* ------------------------------------------------------------------ */
/*  StBuffer                                                           */
/* ------------------------------------------------------------------ */

typedef struct StBuffer {
  StBufferType type;

  /* CPU-visible data pointer.
   * For ST_BUFFER_CPU  : the malloc'd region.
   * For ST_BUFFER_METAL: result of [MTLBuffer contents] — directly
   *                      readable/writable without memcpy.            */
  float *data;

  /* Size in bytes of the usable data region.                          */
  size_t size_bytes;

  /* Number of float elements that fit: size_bytes / sizeof(float).    */
  size_t capacity;

  /* Reference count: allows multiple tensor views on same buffer.
   * Buffer is freed when refcount reaches 0.  Atomic for thread-safety
   * when views are created/destroyed concurrently.                    */
  _Atomic int refcount;

  /* If true, buffer owns `data` and will free() it on release.
   * False for buffers wrapping external pointers.                     */
  bool owns_data;

  /* Opaque pointer to backend-specific resource.
   * ST_BUFFER_CPU  : NULL
   * ST_BUFFER_METAL: __bridge-retained id<MTLBuffer>                  */
  void *_backend_handle;

  /* Opaque pointer to a pending GPU command buffer.
   * Non-NULL when GPU work writing to this buffer has been committed
   * but not yet waited on (async dispatch).  Callers must invoke
   * st_buffer_wait_gpu() / st_tensor_sync() before reading ->data.    */
  void *_async_cmd_buf;

  /* Last completed async GPU command duration, if the backend exposes one.
   * Used by benchmarks/profiling; false when unavailable or not measured. */
  bool _last_gpu_elapsed_valid;
  double _last_gpu_elapsed_ms;
  bool _last_gpu_profile_valid;
  StBufferGpuProfile _last_gpu_profile;

} StBuffer;

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

/// Allocate a CPU buffer of at least `num_floats` elements (zero-filled).
StBuffer *st_buffer_alloc_cpu(size_t num_floats);

/// Allocate a CPU buffer of at least `num_bytes` bytes (zero-filled).
/// For dtype-flexible storage (e.g., bf16 tensors where element size != 4).
/// capacity is set to num_bytes / sizeof(float) (truncated).
StBuffer *st_buffer_alloc_bytes_cpu(size_t num_bytes);

/// Allocate a Metal shared-memory buffer of at least `num_floats` elements.
/// Falls back to CPU allocation if Metal is unavailable.
/// Returns NULL on failure.
StBuffer *st_buffer_alloc_metal(size_t num_floats);

/// Allocate best available buffer type for the current platform.
/// Apple Silicon → Metal; everything else → CPU.
StBuffer *st_buffer_alloc(size_t num_floats);

/// Allocate best available buffer type, sized in raw bytes.
/// For dtype-flexible storage (e.g., bf16 tensors).
StBuffer *st_buffer_alloc_bytes(size_t num_bytes);

/// Wrap an existing CPU pointer.  If `take_ownership` is true the buffer
/// will free(data) on release; otherwise the caller keeps ownership.
StBuffer *st_buffer_from_ptr(float *data, size_t num_floats,
                             bool take_ownership);

/// Increment refcount.  Returns `buf` for convenience.
StBuffer *st_buffer_retain(StBuffer *buf);

/// Decrement refcount; frees backing memory when it reaches 0.
void st_buffer_release(StBuffer *buf);

/* ------------------------------------------------------------------ */
/*  Queries                                                            */
/* ------------------------------------------------------------------ */

/// True when `buf` is GPU-accessible without a copy.
static inline bool st_buffer_is_device(const StBuffer *buf) {
  return buf && buf->type == ST_BUFFER_METAL;
}

/// Returns the raw Metal buffer handle (id<MTLBuffer> cast to void*).
/// NULL for CPU buffers.
static inline void *st_buffer_metal_handle(const StBuffer *buf) {
  return buf ? buf->_backend_handle : NULL;
}

/// Wait for any pending async GPU work on `buf` to complete, then clear
/// the pending marker.  No-op when no work is pending or buf is NULL.
/// Must be called before reading buf->data after an async GPU dispatch.
void st_buffer_wait_gpu(StBuffer *buf);

/// Return the last measured GPU command-buffer duration for this buffer.
/// Returns false when no backend timestamp is available.
bool st_buffer_last_gpu_elapsed_ms(const StBuffer *buf, double *out_ms);

/// Return the last measured host-side GPU orchestration timings.
/// Values are best-effort profiling data for benchmarks.
bool st_buffer_last_gpu_profile(const StBuffer *buf,
                                StBufferGpuProfile *out_profile);

#ifdef __cplusplus
}
#endif

#endif /* ST_BUFFER_H */
