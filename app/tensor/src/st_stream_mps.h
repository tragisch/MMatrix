/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * Shared MPS stream helpers for command-queue and command-buffer handling.
 */

#ifndef ST_STREAM_MPS_H
#define ST_STREAM_MPS_H

#include <stdbool.h>
#include <stddef.h>

typedef struct StBuffer StBuffer;

size_t st_mps_stream_get_commit_every(void);
void st_mps_stream_set_commit_every(size_t every);
bool st_mps_stream_async_defer_enabled(void);
void st_mps_stream_flush(void);

#ifdef __OBJC__
#import <Metal/Metal.h>

id<MTLCommandQueue> st_mps_stream_shared_queue(void);
id<MTLCommandBuffer> st_mps_stream_make_command_buffer(
    id<MTLCommandQueue> queue);
id st_mps_stream_make_mps_command_buffer(id<MTLCommandQueue> queue);
void st_mps_stream_register_pending_buffer(StBuffer *buf);
bool st_mps_stream_finalize_encoded_command_buffer(
    id<MTLCommandBuffer> cmd_buf, id mps_cmd_buf, bool force_commit,
    id<MTLCommandBuffer> *out_pending_cmd_buf);
#endif

#endif  // ST_STREAM_MPS_H
