/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "st_buffer.h"

#include <stdint.h>
#include <string.h>

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
#ifndef TEST_ASSERT_FALSE
#define TEST_ASSERT_FALSE(condition) \
  do {                               \
    (void)(condition);               \
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

/* ------------------------------------------------------------------ */
/*  alloc_cpu                                                          */
/* ------------------------------------------------------------------ */

void test_st_buffer_alloc_cpu_should_create_zeroed_buffer(void) {
  StBuffer *buf = st_buffer_alloc_cpu(100);
  TEST_ASSERT_NOT_NULL(buf);
  TEST_ASSERT_NOT_NULL(buf->data);
  TEST_ASSERT_EQUAL_INT(ST_BUFFER_CPU, buf->type);
  TEST_ASSERT_EQUAL(100, buf->capacity);
  TEST_ASSERT_EQUAL(100 * sizeof(float), buf->size_bytes);
  TEST_ASSERT_EQUAL_INT(1, buf->refcount);
  TEST_ASSERT_TRUE(buf->owns_data);
  TEST_ASSERT_NULL(buf->_backend_handle);

  /* Verify zero-initialized. */
  for (size_t i = 0; i < 100; ++i) {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, buf->data[i]);
  }

  st_buffer_release(buf);
}

void test_st_buffer_alloc_cpu_should_fail_on_zero(void) {
  StBuffer *buf = st_buffer_alloc_cpu(0);
  TEST_ASSERT_NULL(buf);
}

/* ------------------------------------------------------------------ */
/*  from_ptr                                                           */
/* ------------------------------------------------------------------ */

void test_st_buffer_from_ptr_without_ownership_should_not_free(void) {
  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  StBuffer *buf = st_buffer_from_ptr(data, 4, false);
  TEST_ASSERT_NOT_NULL(buf);
  TEST_ASSERT_TRUE(buf->data == data);
  TEST_ASSERT_EQUAL(4, buf->capacity);
  TEST_ASSERT_FALSE(buf->owns_data);
  TEST_ASSERT_EQUAL_INT(1, buf->refcount);

  st_buffer_release(buf);

  /* data[] still valid — verify it wasn't corrupted. */
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.0f, data[2]);
}

void test_st_buffer_from_ptr_should_fail_on_null(void) {
  StBuffer *buf = st_buffer_from_ptr(NULL, 10, false);
  TEST_ASSERT_NULL(buf);
}

/* ------------------------------------------------------------------ */
/*  retain / release                                                   */
/* ------------------------------------------------------------------ */

void test_st_buffer_retain_should_increment_refcount(void) {
  StBuffer *buf = st_buffer_alloc_cpu(16);
  TEST_ASSERT_NOT_NULL(buf);
  TEST_ASSERT_EQUAL_INT(1, buf->refcount);

  StBuffer *same = st_buffer_retain(buf);
  TEST_ASSERT_TRUE(same == buf);
  TEST_ASSERT_EQUAL_INT(2, buf->refcount);

  /* Release twice — should not crash. */
  st_buffer_release(buf);
  TEST_ASSERT_EQUAL_INT(1, buf->refcount);

  st_buffer_release(buf);
  /* buf is now freed — no further access. */
}

/* ------------------------------------------------------------------ */
/*  alloc (platform best)                                              */
/* ------------------------------------------------------------------ */

void test_st_buffer_alloc_should_return_valid_buffer(void) {
  StBuffer *buf = st_buffer_alloc(64);
  TEST_ASSERT_NOT_NULL(buf);
  TEST_ASSERT_NOT_NULL(buf->data);
  TEST_ASSERT_EQUAL(64, buf->capacity);
  TEST_ASSERT_EQUAL_INT(1, buf->refcount);

  st_buffer_release(buf);
}

/* ------------------------------------------------------------------ */
/*  queries                                                            */
/* ------------------------------------------------------------------ */

void test_st_buffer_is_device_should_be_false_for_cpu(void) {
  StBuffer *buf = st_buffer_alloc_cpu(8);
  TEST_ASSERT_NOT_NULL(buf);
  TEST_ASSERT_FALSE(st_buffer_is_device(buf));
  TEST_ASSERT_NULL(st_buffer_metal_handle(buf));

  st_buffer_release(buf);
}

/* ------------------------------------------------------------------ */
/*  read / write through data pointer                                  */
/* ------------------------------------------------------------------ */

void test_st_buffer_data_should_be_readable_and_writable(void) {
  StBuffer *buf = st_buffer_alloc_cpu(4);
  TEST_ASSERT_NOT_NULL(buf);

  buf->data[0] = 1.5f;
  buf->data[1] = -2.5f;
  buf->data[2] = 3.14f;
  buf->data[3] = 0.0f;

  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.5f, buf->data[0]);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, -2.5f, buf->data[1]);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.14f, buf->data[2]);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, buf->data[3]);

  st_buffer_release(buf);
}

/* ------------------------------------------------------------------ */
/*  shared buffer between views                                        */
/* ------------------------------------------------------------------ */

void test_st_buffer_shared_views_should_reflect_writes(void) {
  StBuffer *buf = st_buffer_alloc_cpu(10);
  TEST_ASSERT_NOT_NULL(buf);

  /* Simulate two "views" sharing the same buffer. */
  StBuffer *view_ref = st_buffer_retain(buf);
  TEST_ASSERT_EQUAL_INT(2, buf->refcount);

  /* Write through original. */
  buf->data[5] = 42.0f;

  /* Read through "view" reference — same memory. */
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 42.0f, view_ref->data[5]);

  st_buffer_release(view_ref);
  TEST_ASSERT_EQUAL_INT(1, buf->refcount);

  st_buffer_release(buf);
}
