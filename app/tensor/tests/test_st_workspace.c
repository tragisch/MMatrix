#include "st_workspace.h"

#include <stdint.h>

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
#ifndef TEST_ASSERT_TRUE
#define TEST_ASSERT_TRUE(condition) \
  do {                              \
    (void)(condition);              \
  } while (0)
#endif
#ifndef TEST_ASSERT_FLOAT_WITHIN
#define TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual) \
  do {                                                    \
    (void)(delta);                                        \
    (void)(expected);                                     \
    (void)(actual);                                       \
  } while (0)
#endif

void setUp(void) {}
void tearDown(void) {}

void test_st_workspace_should_grow_reset_and_reuse_storage(void) {
  StWorkspace *ws = st_workspace_get();
  TEST_ASSERT_NOT_NULL(ws);

  float *large = st_workspace_alloc(ws, 300000);
  TEST_ASSERT_NOT_NULL(large);
  large[0] = 1.0f;
  large[299999] = 2.0f;

  st_workspace_reset(ws);

  float *small = st_workspace_calloc(ws, 16);
  TEST_ASSERT_NOT_NULL(small);
  TEST_ASSERT_TRUE(small == large);
  for (size_t i = 0; i < 16; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, small[i]);
  }

  st_workspace_destroy(ws);
}

void test_st_workspace_alloc_should_fail_on_size_overflow(void) {
  StWorkspace *ws = st_workspace_get();
  TEST_ASSERT_NOT_NULL(ws);

  float *ptr = st_workspace_alloc(ws, (SIZE_MAX / sizeof(float)) + 1);
  TEST_ASSERT_NULL(ptr);

  st_workspace_destroy(ws);
}
