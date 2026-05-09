/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_gpu_guard.h — Debug hook: detect CPU API called on pending GPU tensor.
 *
 * Problem: Some CPU-side tensor APIs (st_get, st_clone, st_inplace_add, …)
 * access ->values directly without first calling st_tensor_sync().  When the
 * tensor has uncommitted or in-flight GPU work (_async_cmd_buf != NULL), this
 * produces a silent data race: the CPU reads stale or partially-written data.
 *
 * This header provides ST_ASSERT_NOT_PENDING(t), which:
 *  - Logs a warning and increments a global violation counter.
 *  - Does NOT synchronise automatically (no st_tensor_sync() call).
 *  - Is always active — the fast-path check is a single NULL pointer compare
 *    and adds no measurable overhead when no violation occurs.
 *
 * Usage in a test:
 *   st_gpu_guard_reset_count();
 *   ... trigger the suspect API ...
 *   TEST_ASSERT_EQUAL(0, st_gpu_guard_violation_count());
 */

#ifndef ST_GPU_GUARD_H
#define ST_GPU_GUARD_H

#include "st.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Internal check function — use the macro, not this directly.
void st_gpu_guard_check(const FloatTensor *t, const char *func,
                        const char *file, int line);

/// Returns the total number of pending-GPU violations detected since
/// the last st_gpu_guard_reset_count() call (or program start).
size_t st_gpu_guard_violation_count(void);

/// Reset the violation counter to 0.  Call this in test setUp() or
/// between independent test cases.
void st_gpu_guard_reset_count(void);

/// Assert that tensor t has no pending GPU work before a CPU access.
/// Logs a warning and increments the violation counter on failure.
/// Never syncs automatically.
#define ST_ASSERT_NOT_PENDING(t) \
  st_gpu_guard_check((t), __func__, __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif /* ST_GPU_GUARD_H */

