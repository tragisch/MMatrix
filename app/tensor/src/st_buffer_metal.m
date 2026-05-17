/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_buffer_metal.m — Metal buffer allocation via MTLBuffer (StorageModeShared).
 *
 * Creates GPU-accessible buffers where CPU and GPU share the same
 * physical memory on Apple Silicon — no copy-in / copy-out needed.
 */

#import "st_buffer_metal.h"
#import "sm_mps.h"

#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#import <stdint.h>
#import <stdlib.h>
#import <stdatomic.h>
#import <string.h>
#import <time.h>

#define ST_METAL_POOL_MAX_SLOTS 32u
#define ST_METAL_POOL_MAX_BUFFER_BYTES (64u * 1024u * 1024u)

typedef struct StMetalPoolEntry {
  size_t size_bytes;
  void *handle;
} StMetalPoolEntry;

static _Atomic uint64_t g_alloc_requests = 0u;
static _Atomic uint64_t g_pool_hits = 0u;
static _Atomic uint64_t g_new_allocations = 0u;
static _Atomic uint64_t g_pool_stores = 0u;
static _Atomic uint64_t g_pool_store_drops = 0u;

void st_buffer_metal_allocator_stats_reset_impl(void) {
  atomic_store_explicit(&g_alloc_requests, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_pool_hits, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_new_allocations, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_pool_stores, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_pool_store_drops, 0u, memory_order_relaxed);
}

StBufferMetalAllocatorStats st_buffer_metal_allocator_stats_get_impl(void) {
  StBufferMetalAllocatorStats s;
  s.alloc_requests =
      atomic_load_explicit(&g_alloc_requests, memory_order_relaxed);
  s.pool_hits = atomic_load_explicit(&g_pool_hits, memory_order_relaxed);
  s.new_allocations =
      atomic_load_explicit(&g_new_allocations, memory_order_relaxed);
  s.pool_stores = atomic_load_explicit(&g_pool_stores, memory_order_relaxed);
  s.pool_store_drops =
      atomic_load_explicit(&g_pool_store_drops, memory_order_relaxed);
  return s;
}

static dispatch_queue_t st_metal_pool_queue(void) {
  static dispatch_queue_t q = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    q = dispatch_queue_create("com.mmatrix.st_buffer_metal.pool",
                              DISPATCH_QUEUE_SERIAL);
  });
  return q;
}

static StMetalPoolEntry *st_metal_pool_entries(void) {
  static StMetalPoolEntry entries[ST_METAL_POOL_MAX_SLOTS] = {0};
  return entries;
}

static void *st_metal_pool_take_exact(size_t size_bytes) {
  if (size_bytes == 0u || size_bytes > ST_METAL_POOL_MAX_BUFFER_BYTES) {
    return NULL;
  }

  __block void *found_handle = NULL;
  dispatch_sync(st_metal_pool_queue(), ^{
    StMetalPoolEntry *entries = st_metal_pool_entries();
    for (size_t i = 0; i < ST_METAL_POOL_MAX_SLOTS; ++i) {
      if (entries[i].handle && entries[i].size_bytes == size_bytes) {
        found_handle = entries[i].handle;
        entries[i].handle = NULL;
        entries[i].size_bytes = 0u;
        atomic_fetch_add_explicit(&g_pool_hits, 1u, memory_order_relaxed);
        break;
      }
    }
  });
  return found_handle;
}

static void st_metal_pool_put_handle(void *handle) {
  if (!handle) {
    return;
  }

  id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)handle;
  if (!mtl_buf) {
    id<MTLBuffer> __unused released = (__bridge_transfer id<MTLBuffer>)handle;
    atomic_fetch_add_explicit(&g_pool_store_drops, 1u, memory_order_relaxed);
    return;
  }

  const size_t size_bytes = (size_t)mtl_buf.length;
  if (size_bytes == 0u || size_bytes > ST_METAL_POOL_MAX_BUFFER_BYTES) {
    id<MTLBuffer> __unused released = (__bridge_transfer id<MTLBuffer>)handle;
    atomic_fetch_add_explicit(&g_pool_store_drops, 1u, memory_order_relaxed);
    return;
  }

  __block bool stored = false;
  dispatch_sync(st_metal_pool_queue(), ^{
    StMetalPoolEntry *entries = st_metal_pool_entries();
    for (size_t i = 0; i < ST_METAL_POOL_MAX_SLOTS; ++i) {
      if (!entries[i].handle) {
        entries[i].handle = handle;
        entries[i].size_bytes = size_bytes;
        stored = true;
        atomic_fetch_add_explicit(&g_pool_stores, 1u, memory_order_relaxed);
        return;
      }
    }
  });

  if (!stored) {
    id<MTLBuffer> __unused released = (__bridge_transfer id<MTLBuffer>)handle;
    atomic_fetch_add_explicit(&g_pool_store_drops, 1u, memory_order_relaxed);
  }
}

static double st_metal_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

StBuffer *st_buffer_alloc_metal_impl(size_t num_floats) {
  if (num_floats == 0) {
    return NULL;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)mps_get_shared_device();
  if (!device) {
    return NULL;
  }

  if (num_floats > SIZE_MAX / sizeof(float)) {
    return NULL;
  }

  const size_t size_bytes = num_floats * sizeof(float);
  atomic_fetch_add_explicit(&g_alloc_requests, 1u, memory_order_relaxed);

  void *reused_handle = st_metal_pool_take_exact(size_bytes);
  id<MTLBuffer> mtl_buf = reused_handle
                              ? (__bridge id<MTLBuffer>)reused_handle
                              : [device newBufferWithLength:size_bytes
                                                  options:MTLResourceStorageModeShared];
  if (!mtl_buf) {
    return NULL;
  }
  if (!reused_handle) {
    atomic_fetch_add_explicit(&g_new_allocations, 1u, memory_order_relaxed);
  }

  /* Zero-fill to match st_buffer_alloc_cpu() behaviour. */
  memset(mtl_buf.contents, 0, size_bytes);

  StBuffer *buf = (StBuffer *)calloc(1, sizeof(StBuffer));
  if (!buf) {
    /* mtl_buf released by ARC when leaving scope. */
    return NULL;
  }

  buf->type = ST_BUFFER_METAL;
  buf->data = (float *)mtl_buf.contents;
  buf->size_bytes = size_bytes;
  buf->capacity = num_floats;
  buf->refcount = 1;
  buf->owns_data = true;

  /* Bridge-retain: ARC won't release the MTLBuffer as long as we
   * hold this void* — we release it in st_buffer_release_metal_handle(). */
  buf->_backend_handle = reused_handle ? reused_handle
                                       : (__bridge_retained void *)mtl_buf;

  return buf;
}

void st_buffer_release_metal_handle(void *handle) {
  if (!handle) {
    return;
  }
  st_metal_pool_put_handle(handle);
}

void st_buffer_metal_discard_pending(StBuffer *buf) {
  if (!buf || !buf->_async_cmd_buf) {
    return;
  }
  void *handle = buf->_async_cmd_buf;
  buf->_async_cmd_buf = NULL;
  /* Transfer ownership to ARC and release without waiting. */
  id<MTLCommandBuffer> __unused cmdBuf =
      (__bridge_transfer id<MTLCommandBuffer>)handle;
}

void st_buffer_metal_discard_handle(void *cmd_handle) {
  if (!cmd_handle) {
    return;
  }
  id<MTLCommandBuffer> __unused cmdBuf =
      (__bridge_transfer id<MTLCommandBuffer>)cmd_handle;
}

void st_buffer_metal_wait(StBuffer *buf, StBufferWaitReason reason) {
  if (!buf || !buf->_async_cmd_buf) {
    return;
  }
  void *handle = buf->_async_cmd_buf;
  buf->_async_cmd_buf = NULL;
  st_buffer_metal_wait_handle(handle, buf, reason);
}

void st_buffer_metal_wait_handle(void *cmd_handle, StBuffer *profile_buf,
                                 StBufferWaitReason reason) {
  if (!cmd_handle) {
    return;
  }

  StBuffer *buf = profile_buf;
  if (!buf) {
    id<MTLCommandBuffer> cmdBuf = (__bridge_transfer id<MTLCommandBuffer>)cmd_handle;
    [cmdBuf waitUntilCompleted];
    return;
  }

  buf->_last_gpu_elapsed_valid = false;
  buf->_last_gpu_elapsed_ms = 0.0;
  /* Transfer ownership to ARC so the command buffer is released after wait. */
  id<MTLCommandBuffer> cmdBuf = (__bridge_transfer id<MTLCommandBuffer>)cmd_handle;
  const double wait_start_ms = st_metal_now_ms();
  [cmdBuf waitUntilCompleted];
  const double wait_end_ms = st_metal_now_ms();
  const double wait_ms = wait_end_ms - wait_start_ms;
  buf->_last_gpu_profile.sync_wait_ms = wait_ms;
  buf->_last_gpu_profile.sync_wait_prewrite_ms = 0.0;
  buf->_last_gpu_profile.sync_wait_boundary_ms = 0.0;
  if (reason == ST_BUFFER_WAIT_REASON_PREWRITE) {
    buf->_last_gpu_profile.sync_wait_prewrite_ms = wait_ms;
  } else if (reason == ST_BUFFER_WAIT_REASON_BOUNDARY) {
    buf->_last_gpu_profile.sync_wait_boundary_ms = wait_ms;
  }
  buf->_last_gpu_profile_valid = true;
  if (cmdBuf.status == MTLCommandBufferStatusCompleted &&
      cmdBuf.GPUEndTime > cmdBuf.GPUStartTime) {
    buf->_last_gpu_elapsed_ms =
        (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
    buf->_last_gpu_elapsed_valid = true;
  }
  /* cmdBuf released by ARC here. */
}

bool st_buffer_metal_schedule_release(void *cmd_handle, void *metal_handle) {
  if (!cmd_handle || !metal_handle) {
    return false;
  }

  void *single[1] = {cmd_handle};
  return st_buffer_metal_schedule_release_many(single, 1u, metal_handle);
}

bool st_buffer_metal_schedule_release_many(void **cmd_handles,
                                           size_t cmd_count,
                                           void *metal_handle) {
  if (!metal_handle || !cmd_handles || cmd_count == 0u) {
    return false;
  }

  void **cmd_copy = (void **)calloc(cmd_count, sizeof(void *));
  if (!cmd_copy) {
    return false;
  }
  memcpy(cmd_copy, cmd_handles, cmd_count * sizeof(void *));

  void *metal_handle_for_block = metal_handle;
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
    for (size_t i = 0; i < cmd_count; ++i) {
      void *h = cmd_copy[i];
      if (!h) {
        continue;
      }
      id<MTLCommandBuffer> released_cmd =
          (__bridge_transfer id<MTLCommandBuffer>)h;
      if (released_cmd) {
        [released_cmd waitUntilCompleted];
      }
    }
    free(cmd_copy);
    st_metal_pool_put_handle(metal_handle_for_block);
  });
  return true;
}
