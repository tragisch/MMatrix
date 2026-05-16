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
#import <stdint.h>
#import <string.h>
#import <time.h>

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

  id<MTLBuffer> mtl_buf =
      [device newBufferWithLength:size_bytes
                          options:MTLResourceStorageModeShared];
  if (!mtl_buf) {
    return NULL;
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
  buf->_backend_handle = (__bridge_retained void *)mtl_buf;

  return buf;
}

void st_buffer_release_metal_handle(void *handle) {
  if (!handle) {
    return;
  }
  /* Transfer ownership back to ARC, which will release the MTLBuffer. */
  id<MTLBuffer> __unused mtl_buf = (__bridge_transfer id<MTLBuffer>)handle;
  /* mtl_buf goes out of scope here → ARC releases it. */
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

void st_buffer_metal_wait(StBuffer *buf, StBufferWaitReason reason) {
  if (!buf || !buf->_async_cmd_buf) {
    return;
  }
  void *handle = buf->_async_cmd_buf;
  buf->_async_cmd_buf = NULL;
  buf->_last_gpu_elapsed_valid = false;
  buf->_last_gpu_elapsed_ms = 0.0;
  /* Transfer ownership to ARC so the command buffer is released after wait. */
  id<MTLCommandBuffer> cmdBuf = (__bridge_transfer id<MTLCommandBuffer>)handle;
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
