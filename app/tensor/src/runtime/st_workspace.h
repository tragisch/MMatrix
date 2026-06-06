/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_workspace.h — Thread-local bump allocator for temporary buffers.
 *
 * Eliminates per-call malloc/free in hot paths (im2col, GEMM temps, etc.).
 * The workspace grows once to the high-water mark and is reused via reset().
 * Memory is 64-byte aligned for SIMD compatibility.
 */

#ifndef ST_WORKSPACE_H
#define ST_WORKSPACE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StWorkspace StWorkspace;

/// Get the thread-local workspace (created on first call).
StWorkspace *st_workspace_get(void);

/// Bump-allocate `num_floats` contiguous floats from the workspace.
/// Returns a 64-byte-aligned pointer.  The memory is NOT zero-filled.
/// Returns NULL on failure (grow failed).
float *st_workspace_alloc(StWorkspace *ws, size_t num_floats);

/// Same as st_workspace_alloc but zero-fills the returned region.
float *st_workspace_calloc(StWorkspace *ws, size_t num_floats);

/// Reset the bump pointer to the beginning.  Does not free memory.
/// Call this at the start of each forward/backward pass.
void st_workspace_reset(StWorkspace *ws);

/// Free the workspace buffer entirely.  Mainly for leak-free shutdown.
void st_workspace_destroy(StWorkspace *ws);

#ifdef __cplusplus
}
#endif

#endif /* ST_WORKSPACE_H */