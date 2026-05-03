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

/// Release a bridge-retained MTLBuffer handle.
/// Called by st_buffer_release() when _backend_handle is non-NULL.
void st_buffer_release_metal_handle(void *handle);

/// Wait for a pending GPU command buffer (bridge-retained id<MTLCommandBuffer>)
/// stored in buf->_async_cmd_buf, then bridge-transfer ownership back to ARC.
/// No-op if buf->_async_cmd_buf is NULL.
void st_buffer_metal_wait(StBuffer *buf);

#ifdef __cplusplus
}
#endif

#endif /* ST_BUFFER_METAL_H */
