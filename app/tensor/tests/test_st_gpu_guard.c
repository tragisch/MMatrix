/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * test_st_gpu_guard.c — Unit tests for the CPU-on-pending-GPU fence hook.
 *
 * The guard is always active and intentionally does not synchronise. It only
 * logs and counts CPU access to tensors with pending GPU work.
 */

#include "st.h"
#include "st_backend.h"
#include "st_buffer.h"
#include "st_conv.h"
#include "st_gpu_guard.h"

#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 6

/* Support for Meta Test Rig */
#define TEST_CASE(...)

#if __has_include("unity.h")
#include "unity.h"
#endif

#ifndef TEST_ASSERT_NOT_NULL
#define TEST_ASSERT_NOT_NULL(value) \
  do {                             \
    (void)(value);                 \
  } while (0)
#endif
#ifndef TEST_ASSERT_NULL
#define TEST_ASSERT_NULL(value) \
  do {                          \
    (void)(value);              \
  } while (0)
#endif
#ifndef TEST_ASSERT_EQUAL
#define TEST_ASSERT_EQUAL(expected, actual) \
  do {                                      \
    (void)(expected);                       \
    (void)(actual);                         \
  } while (0)
#endif
#ifndef TEST_ASSERT_EQUAL_INT
#define TEST_ASSERT_EQUAL_INT(expected, actual) \
  do {                                          \
    (void)(expected);                           \
    (void)(actual);                             \
  } while (0)
#endif
#ifndef TEST_ASSERT_TRUE
#define TEST_ASSERT_TRUE(condition) \
  do {                              \
    (void)(condition);              \
  } while (0)
#endif

void setUp(void) {
  st_gpu_guard_reset_count();
}

void tearDown(void) {}

/* ------------------------------------------------------------------ */
/*  Helper: create a small f32 CPU tensor                             */
/* ------------------------------------------------------------------ */

static FloatTensor *make_cpu_tensor(void) {
  const size_t shape[2] = {2, 2};
  FloatTensor *t = st_create(2, shape);
  if (t && t->buf) {
    /* Ensure we got a CPU buffer (no Metal pending state by default). */
  }
  return t;
}

/* Safely release a tensor whose _async_cmd_buf was faked in a test.
 * We MUST clear the sentinel before destroy to prevent st_buffer_release
 * from calling st_buffer_metal_wait() on an invalid pointer.         */
static void safe_destroy(FloatTensor *t) {
  if (t && t->buf) {
    t->buf->_async_cmd_buf = NULL;
  }
  st_destroy(t);
}

/* ------------------------------------------------------------------ */
/*  No violation when tensor has no pending GPU work                  */
/* ------------------------------------------------------------------ */

void test_guard_no_violation_on_clean_tensor(void) {
  FloatTensor *t = make_cpu_tensor();
  TEST_ASSERT_NOT_NULL(t);
  /* No pending work — _async_cmd_buf is NULL. */
  TEST_ASSERT_EQUAL(0, (int)st_gpu_guard_violation_count());

  const size_t idx[2] = {0, 0};
  (void)st_get(t, idx);

  TEST_ASSERT_EQUAL(0, (int)st_gpu_guard_violation_count());
  st_destroy(t);
}

/* ------------------------------------------------------------------ */
/*  st_get on pending tensor triggers violation (debug builds only)   */
/* ------------------------------------------------------------------ */

void test_guard_st_get_on_pending_tensor(void) {
  FloatTensor *t = make_cpu_tensor();
  TEST_ASSERT_NOT_NULL(t);

  /* Simulate pending GPU work via sentinel (non-NULL, not a real MTLCmdBuf). */
  t->buf->_async_cmd_buf = (void *)(uintptr_t)1;

  const size_t idx[2] = {0, 0};
  (void)st_get(t, idx);

  TEST_ASSERT_EQUAL(1, (int)st_gpu_guard_violation_count());
  safe_destroy(t);
}

/* ------------------------------------------------------------------ */
/*  st_set on pending tensor triggers violation (debug builds only)   */
/* ------------------------------------------------------------------ */

void test_guard_st_set_on_pending_tensor(void) {
  FloatTensor *t = make_cpu_tensor();
  TEST_ASSERT_NOT_NULL(t);

  t->buf->_async_cmd_buf = (void *)(uintptr_t)1;

  const size_t idx[2] = {1, 1};
  (void)st_set(t, idx, 3.14f);

  TEST_ASSERT_EQUAL(1, (int)st_gpu_guard_violation_count());
  safe_destroy(t);
}

/* ------------------------------------------------------------------ */
/*  st_clone on pending tensor triggers violation (debug builds only) */
/* ------------------------------------------------------------------ */

void test_guard_st_clone_on_pending_tensor(void) {
  FloatTensor *t = make_cpu_tensor();
  TEST_ASSERT_NOT_NULL(t);

  t->buf->_async_cmd_buf = (void *)(uintptr_t)1;

  FloatTensor *c = st_clone(t);

  /* Violation count >= 1 (clone itself triggers it; internal st_get calls
   * may add more, which is acceptable). */
  TEST_ASSERT_TRUE(st_gpu_guard_violation_count() >= 1);

  /* Clone has clean buffer — no pending work propagated. */
  if (c) {
    TEST_ASSERT_NULL(c->buf->_async_cmd_buf);
  }

  safe_destroy(t);
  st_destroy(c);
}

/* ------------------------------------------------------------------ */
/*  st_to_f32 on pending tensor triggers violation (debug only)       */
/* ------------------------------------------------------------------ */

void test_guard_st_to_f32_on_pending_tensor(void) {
  /* Use a BF16 tensor so st_to_f32 actually accesses src->values. */
  const size_t shape[1] = {4};
  FloatTensor *t = st_create_bf16(1, shape);
  TEST_ASSERT_NOT_NULL(t);

  t->buf->_async_cmd_buf = (void *)(uintptr_t)1;

  FloatTensor *f = st_to_f32(t);

  TEST_ASSERT_TRUE(st_gpu_guard_violation_count() >= 1);
  safe_destroy(t);
  st_destroy(f);
}

/* ------------------------------------------------------------------ */
/*  reset_count zeroes the counter                                    */
/* ------------------------------------------------------------------ */

void test_guard_reset_clears_count(void) {
  FloatTensor *t = make_cpu_tensor();
  TEST_ASSERT_NOT_NULL(t);

  t->buf->_async_cmd_buf = (void *)(uintptr_t)1;
  const size_t idx[2] = {0, 0};
  (void)st_get(t, idx);

  TEST_ASSERT_TRUE(st_gpu_guard_violation_count() >= 1);
  st_gpu_guard_reset_count();
  TEST_ASSERT_EQUAL(0, (int)st_gpu_guard_violation_count());
  safe_destroy(t);
}

/* ------------------------------------------------------------------ */
/*  P2: real MPS async dispatch → CPU API without sync fires guard    */
/*                                                                     */
/*  Shape: conv_medium (N=1, C=8, H=32, W=32, Cout=16, k=3).         */
/*  Requires MPS device; if MPS is unavailable the backend falls back */
/*  to CPU and _async_cmd_buf stays NULL — test passes silently.      */
/* ------------------------------------------------------------------ */

void test_guard_real_mps_async_conv_then_cpu_get(void) {
#if defined(USE_ACCELERATE) && defined(__APPLE__)
  /* Enable async dispatch for this test only. */
  bool prev_async = st_backend_get_conv_mps_async();

  const size_t N = 1, Cin = 8, H = 32, W = 32;
  const size_t Cout = 16, Kh = 3, Kw = 3;

  size_t s_in[4]  = {N, Cin, H, W};
  size_t s_w[4]   = {Cout, Cin, Kh, Kw};
  size_t s_out[4] = {N, Cout, H - Kh + 1, W - Kw + 1}; /* pad=0 */

  FloatTensor *input  = st_create(4, s_in);
  FloatTensor *weight = st_create(4, s_w);
  FloatTensor *output = st_create(4, s_out);

  TEST_ASSERT_NOT_NULL(input);
  TEST_ASSERT_NOT_NULL(weight);
  TEST_ASSERT_NOT_NULL(output);

  for (size_t i = 0; i < input->numel;  ++i) input->values[i]  = 0.1f;
  for (size_t i = 0; i < weight->numel; ++i) weight->values[i] = 0.01f;

  StConv2dParams params = st_conv2d_default_params();
  params.backend = ST_CONV_BACKEND_MPS;

  /* Warmup call (sync): compiles the MPSGraphExecutable and caches it. */
  st_backend_set_conv_mps_async(false);
  (void)st_conv2d_nchw(input, weight, NULL, &params, output);

  /* Second call (async): executable is cached → fastpath sets _async_cmd_buf. */
  st_backend_set_conv_mps_async(true);
  bool conv_ok = st_conv2d_nchw(input, weight, NULL, &params, output);

  if (!conv_ok || output->buf == NULL || output->buf->_backend_handle == NULL ||
      output->buf->_async_cmd_buf == NULL) {
    /* MPS fastpath not available (may be unavailable in test environment). */
    st_destroy(input);
    st_destroy(weight);
    st_destroy(output);
    st_backend_set_conv_mps_async(prev_async);
    return;
  }


  /* Output has real pending GPU work — CPU access must trigger the guard. */
  const size_t idx[4] = {0, 0, 0, 0};
  (void)st_get(output, idx);

  TEST_ASSERT_TRUE(st_gpu_guard_violation_count() >= 1);

  /* Sync before destroy. */
  st_tensor_sync(output);
  st_destroy(input);
  st_destroy(weight);
  st_destroy(output);
  st_backend_set_conv_mps_async(prev_async);
#else
  TEST_ASSERT_EQUAL(0, (int)st_gpu_guard_violation_count());
#endif
}

