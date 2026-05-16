/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_buffer_metal.h — C-callable bridge for Metal buffer allocation.
 *
 * Internal header — only included by st_buffer.c under
 * #if defined(USE_ACCELERATE) && defined(__APPLE__).
 */

#ifndef ST_BUFFER_METAL_H
#define ST_BUFFER_METAL_H

#include "st_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Allocate a Metal shared-memory buffer (MTLResourceStorageModeShared).
/// buf->data  = [MTLBuffer contents]   (CPU-visible pointer)
/// buf->_backend_handle = bridge-retained id<MTLBuffer>
/// Returns NULL on failure (no Metal device available, OOM, etc.).
StBuffer *st_buffer_alloc_metal_impl(size_t num_floats);

typedef enum StBufferWaitReason {
	ST_BUFFER_WAIT_REASON_BOUNDARY = 0,
	ST_BUFFER_WAIT_REASON_PREWRITE = 1,
	ST_BUFFER_WAIT_REASON_RELEASE = 2,
} StBufferWaitReason;

/// Release a bridge-retained MTLBuffer handle.
/// Called by st_buffer_release() when _backend_handle is non-NULL.
void st_buffer_release_metal_handle(void *handle);

/// Drop tracking of a pending GPU command buffer without waiting.
/// Safe when subsequent work is enqueued on the same command queue and
/// boundary sync waits on a newer command buffer (queue ordering guarantees
/// completion of older command buffers first).
void st_buffer_metal_discard_pending(StBuffer *buf);

/// Release one bridge-retained command-buffer handle without waiting.
void st_buffer_metal_discard_handle(void *cmd_handle);

/// Wait for a pending GPU command buffer (bridge-retained id<MTLCommandBuffer>)
/// stored in buf->_async_cmd_buf, then bridge-transfer ownership back to ARC.
/// No-op if buf->_async_cmd_buf is NULL.
void st_buffer_metal_wait(StBuffer *buf, StBufferWaitReason reason);

/// Wait one bridge-retained command-buffer handle, then release it.
/// Writes best-effort timing/profile into `profile_buf` when non-NULL.
void st_buffer_metal_wait_handle(void *cmd_handle, StBuffer *profile_buf,
								 StBufferWaitReason reason);

/// Schedule asynchronous release of both handles once cmd buffer completes.
/// cmd_handle must be a bridge-retained id<MTLCommandBuffer> and
/// metal_handle must be a bridge-retained id<MTLBuffer>.
/// Returns true when completion-handler registration succeeded.
bool st_buffer_metal_schedule_release(void *cmd_handle, void *metal_handle);

/// Schedule asynchronous wait+release for many command buffers and one
/// retained MTLBuffer handle. `cmd_handles` entries must be bridge-retained
/// id<MTLCommandBuffer> handles.
bool st_buffer_metal_schedule_release_many(void **cmd_handles,
										   size_t cmd_count,
										   void *metal_handle);

#ifdef __cplusplus
}
#endif

#endif /* ST_BUFFER_METAL_H */
