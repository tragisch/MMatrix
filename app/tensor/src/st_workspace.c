/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_workspace.c — Thread-local bump allocator implementation.
 */

#include "st_workspace.h"

#include <log.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* 64-byte alignment for SIMD (NEON, AVX-512). */
#define ST_WS_ALIGN 64

/* Initial capacity: 1 MiB — covers most single-layer Conv temps. */
#define ST_WS_INITIAL_BYTES (1u << 20)

struct StWorkspace {
  char *data;       /* base pointer (64-byte aligned) */
  size_t capacity;  /* total bytes available */
  size_t used;      /* current bump offset in bytes */
};

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

static bool st_workspace_bytes_for_floats(size_t num_floats,
                                          size_t *out_bytes) {
  if (out_bytes == NULL || num_floats == 0 ||
      num_floats > SIZE_MAX / sizeof(float)) {
    return false;
  }

  *out_bytes = num_floats * sizeof(float);
  return true;
}

static bool align_up(size_t n, size_t align, size_t *out_aligned) {
  if (out_aligned == NULL || align == 0 || n > SIZE_MAX - (align - 1)) {
    return false;
  }

  *out_aligned = (n + align - 1) & ~(align - 1);
  return true;
}

/// Grow workspace to hold at least `min_capacity` bytes.
static bool st_workspace_grow(StWorkspace *ws, size_t min_capacity) {
  size_t new_cap = ws->capacity ? ws->capacity : ST_WS_INITIAL_BYTES;
  while (new_cap < min_capacity) {
    if (new_cap > SIZE_MAX / 2) {
      new_cap = min_capacity;
      break;
    }
    new_cap *= 2;
  }

  void *raw = NULL;
  if (posix_memalign(&raw, ST_WS_ALIGN, new_cap) != 0) {
    log_error("Error: st_workspace_grow posix_memalign(%zu) failed.", new_cap);
    return false;
  }

  /* Preserve existing data up to ws->used (needed when growing mid-pass). */
  if (ws->data && ws->used > 0) {
    memcpy(raw, ws->data, ws->used);
  }
  free(ws->data);

  ws->data = (char *)raw;
  ws->capacity = new_cap;
  return true;
}

/* ------------------------------------------------------------------ */
/*  Thread-local singleton                                             */
/* ------------------------------------------------------------------ */

static _Thread_local StWorkspace *t_workspace = NULL;

StWorkspace *st_workspace_get(void) {
  if (t_workspace) {
    return t_workspace;
  }

  t_workspace = (StWorkspace *)calloc(1, sizeof(StWorkspace));
  if (!t_workspace) {
    log_error("Error: st_workspace_get allocation failed.");
    return NULL;
  }
  return t_workspace;
}

/* ------------------------------------------------------------------ */
/*  Bump allocator                                                     */
/* ------------------------------------------------------------------ */

float *st_workspace_alloc(StWorkspace *ws, size_t num_floats) {
  if (!ws || num_floats == 0) {
    return NULL;
  }

  size_t bytes_needed = 0;
  if (!st_workspace_bytes_for_floats(num_floats, &bytes_needed)) {
    log_error("Error: st_workspace_alloc size overflow.");
    return NULL;
  }

  size_t aligned_offset = 0;
  if (!align_up(ws->used, ST_WS_ALIGN, &aligned_offset)) {
    log_error("Error: st_workspace_alloc alignment overflow.");
    return NULL;
  }

  if (bytes_needed > SIZE_MAX - aligned_offset) {
    log_error("Error: st_workspace_alloc usage overflow.");
    return NULL;
  }
  const size_t new_used = aligned_offset + bytes_needed;

  if (new_used > ws->capacity) {
    if (!st_workspace_grow(ws, new_used)) {
      return NULL;
    }
  }

  float *ptr = (float *)(ws->data + aligned_offset);
  ws->used = new_used;
  return ptr;
}

float *st_workspace_calloc(StWorkspace *ws, size_t num_floats) {
  float *ptr = st_workspace_alloc(ws, num_floats);
  if (ptr) {
    size_t bytes_needed = 0;
    if (!st_workspace_bytes_for_floats(num_floats, &bytes_needed)) {
      return NULL;
    }
    memset(ptr, 0, bytes_needed);
  }
  return ptr;
}

/* ------------------------------------------------------------------ */
/*  Reset / Destroy                                                    */
/* ------------------------------------------------------------------ */

void st_workspace_reset(StWorkspace *ws) {
  if (ws) {
    ws->used = 0;
  }
}

void st_workspace_destroy(StWorkspace *ws) {
  if (!ws) {
    return;
  }
  free(ws->data);
  free(ws);
  if (t_workspace == ws) {
    t_workspace = NULL;
  }
}
