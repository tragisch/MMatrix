/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_gpu_guard.c — Implementation of the CPU-on-pending-GPU fence hook.
 */

#include "st_gpu_guard.h"
#include "st_buffer.h"

#include <log.h>
#include <stdatomic.h>

static _Atomic size_t _st_guard_violation_count = 0;

void st_gpu_guard_check(const FloatTensor *t, const char *func,
                        const char *file, int line) {
  if (t && t->buf && t->buf->_async_cmd_buf != NULL) {
    atomic_fetch_add(&_st_guard_violation_count, 1u);
    log_warn(
        "[ST_GPU_GUARD] CPU API '%s' called on pending GPU tensor at %s:%d"
        " — call st_tensor_sync() first.",
        func, file, line);
  }
}

size_t st_gpu_guard_violation_count(void) {
  return atomic_load(&_st_guard_violation_count);
}

void st_gpu_guard_reset_count(void) {
  atomic_store(&_st_guard_violation_count, 0u);
}
