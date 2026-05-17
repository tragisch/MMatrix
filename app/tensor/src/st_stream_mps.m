/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * Shared MPS stream helpers for command-queue and command-buffer handling.
 */

#import "st_stream_mps.h"

#import "st_buffer.h"
#import "sm_mps.h"

#import <Foundation/Foundation.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <objc/message.h>

static _Atomic size_t g_stream_commit_every = 1u;
static _Atomic bool g_stream_commit_every_initialized = false;
static _Atomic bool g_stream_async_defer = false;
static _Atomic bool g_stream_async_defer_initialized = false;

static _Thread_local CFTypeRef g_tls_cmd_handle = NULL;
static _Thread_local CFTypeRef g_tls_mps_cmd_handle = NULL;
static _Thread_local size_t g_tls_encode_count = 0u;
static _Thread_local StBuffer *g_tls_pending_buffers[ST_BUFFER_PENDING_CMDS_MAX];
static _Thread_local size_t g_tls_pending_buffer_count = 0u;

static size_t st_mps_stream_parse_commit_every_env(void) {
  const char *v = getenv("MMATRIX_ST_STREAM_COMMIT_EVERY");
  if (!v || v[0] == '\0') {
    return 1u;
  }
  char *end = NULL;
  const unsigned long parsed = strtoul(v, &end, 10);
  if (end == v || (end && *end != '\0') || parsed == 0ul) {
    return 1u;
  }
  return (size_t)parsed;
}

static bool st_mps_stream_parse_bool_env(const char *name, bool fallback) {
  const char *v = getenv(name);
  if (!v || v[0] == '\0') {
    return fallback;
  }

  if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 ||
      strcmp(v, "TRUE") == 0 || strcmp(v, "yes") == 0 ||
      strcmp(v, "YES") == 0 || strcmp(v, "on") == 0 ||
      strcmp(v, "ON") == 0) {
    return true;
  }

  if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 ||
      strcmp(v, "FALSE") == 0 || strcmp(v, "no") == 0 ||
      strcmp(v, "NO") == 0 || strcmp(v, "off") == 0 ||
      strcmp(v, "OFF") == 0) {
    return false;
  }

  return fallback;
}

static inline id<MTLCommandBuffer> st_mps_stream_tls_cmd(void) {
  return g_tls_cmd_handle
             ? (__bridge id<MTLCommandBuffer>)g_tls_cmd_handle
             : nil;
}

static inline id st_mps_stream_tls_mps_cmd(void) {
  return g_tls_mps_cmd_handle ? (__bridge id)g_tls_mps_cmd_handle : nil;
}

static void st_mps_stream_tls_set_cmd(id<MTLCommandBuffer> cmd_buf) {
  if (g_tls_cmd_handle) {
    CFRelease(g_tls_cmd_handle);
    g_tls_cmd_handle = NULL;
  }
  if (cmd_buf) {
    g_tls_cmd_handle = CFBridgingRetain(cmd_buf);
  }
}

static void st_mps_stream_tls_set_mps_cmd(id mps_cmd_buf) {
  if (g_tls_mps_cmd_handle) {
    CFRelease(g_tls_mps_cmd_handle);
    g_tls_mps_cmd_handle = NULL;
  }
  if (mps_cmd_buf) {
    g_tls_mps_cmd_handle = CFBridgingRetain(mps_cmd_buf);
  }
}

static void st_mps_stream_tls_reset_state(void) {
  st_mps_stream_tls_set_cmd(nil);
  st_mps_stream_tls_set_mps_cmd(nil);
  g_tls_encode_count = 0u;
  g_tls_pending_buffer_count = 0u;
}

static void st_mps_stream_init_commit_every_once(void) {
  if (atomic_load_explicit(&g_stream_commit_every_initialized,
                           memory_order_acquire)) {
    return;
  }
  const size_t every = st_mps_stream_parse_commit_every_env();
  atomic_store_explicit(&g_stream_commit_every, every, memory_order_release);
  atomic_store_explicit(&g_stream_commit_every_initialized, true,
                        memory_order_release);
}

static void st_mps_stream_init_async_defer_once(void) {
  if (atomic_load_explicit(&g_stream_async_defer_initialized,
                           memory_order_acquire)) {
    return;
  }
  const bool enabled =
      st_mps_stream_parse_bool_env("MMATRIX_ST_STREAM_DEFER_ASYNC", false);
  atomic_store_explicit(&g_stream_async_defer, enabled, memory_order_release);
  atomic_store_explicit(&g_stream_async_defer_initialized, true,
                        memory_order_release);
}

size_t st_mps_stream_get_commit_every(void) {
  st_mps_stream_init_commit_every_once();
  return atomic_load_explicit(&g_stream_commit_every, memory_order_acquire);
}

void st_mps_stream_set_commit_every(size_t every) {
  if (every == 0u) {
    every = 1u;
  }
  atomic_store_explicit(&g_stream_commit_every, every, memory_order_release);
  atomic_store_explicit(&g_stream_commit_every_initialized, true,
                        memory_order_release);
}

bool st_mps_stream_async_defer_enabled(void) {
  st_mps_stream_init_async_defer_once();
  return atomic_load_explicit(&g_stream_async_defer, memory_order_acquire);
}

id<MTLCommandQueue> st_mps_stream_shared_queue(void) {
  return (__bridge id<MTLCommandQueue>)mps_get_shared_command_queue();
}

id<MTLCommandBuffer> st_mps_stream_make_command_buffer(
    id<MTLCommandQueue> queue) {
  if (!queue) {
    return nil;
  }

  st_mps_stream_init_commit_every_once();
  const size_t commit_every =
      atomic_load_explicit(&g_stream_commit_every, memory_order_acquire);
  if (commit_every > 1u) {
    id<MTLCommandBuffer> tls_cmd = st_mps_stream_tls_cmd();
    if (tls_cmd) {
      return tls_cmd;
    }
  }

  if ([queue respondsToSelector:@selector(commandBufferWithDescriptor:)]) {
    MTLCommandBufferDescriptor *cb_desc =
        [[MTLCommandBufferDescriptor alloc] init];
    id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithDescriptor:cb_desc];
    if (cmd_buf) {
      if (commit_every > 1u) {
        st_mps_stream_tls_set_cmd(cmd_buf);
      }
      return cmd_buf;
    }
  }

  id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
  if (cmd_buf && commit_every > 1u) {
    st_mps_stream_tls_set_cmd(cmd_buf);
  }
  return cmd_buf;
}

id st_mps_stream_make_mps_command_buffer(id<MTLCommandQueue> queue) {
  if (!queue) {
    return nil;
  }

  const size_t commit_every = st_mps_stream_get_commit_every();
  if (commit_every > 1u) {
    id tls_mps_cmd = st_mps_stream_tls_mps_cmd();
    if (tls_mps_cmd) {
      return tls_mps_cmd;
    }
  }

  Class mps_cb_class = NSClassFromString(@"MPSCommandBuffer");
  if (!mps_cb_class ||
      ![mps_cb_class respondsToSelector:@selector(commandBufferFromCommandQueue:)]) {
    return nil;
  }

  id (*msg_send_cb)(id, SEL, id) = (id (*)(id, SEL, id))objc_msgSend;
  id mps_cmd_buf = msg_send_cb(
      mps_cb_class, @selector(commandBufferFromCommandQueue:), queue);
  if (mps_cmd_buf && commit_every > 1u) {
    st_mps_stream_tls_set_mps_cmd(mps_cmd_buf);
  }
  return mps_cmd_buf;
}

void st_mps_stream_register_pending_buffer(StBuffer *buf) {
  if (!buf) {
    return;
  }
  for (size_t i = 0; i < g_tls_pending_buffer_count; ++i) {
    if (g_tls_pending_buffers[i] == buf) {
      return;
    }
  }

  if (g_tls_pending_buffer_count < ST_BUFFER_PENDING_CMDS_MAX) {
    g_tls_pending_buffers[g_tls_pending_buffer_count++] = buf;
    return;
  }

  g_tls_pending_buffers[g_tls_pending_buffer_count - 1u] = buf;
}

static bool st_mps_stream_commit_internal(
    id<MTLCommandBuffer> cmd_buf, id mps_cmd_buf,
    id<MTLCommandBuffer> *out_pending_cmd_buf) {
  if (!cmd_buf) {
    return false;
  }

  if (mps_cmd_buf && [mps_cmd_buf respondsToSelector:@selector(commit)]) {
    void (*msg_send_void)(id, SEL) = (void (*)(id, SEL))objc_msgSend;
    msg_send_void(mps_cmd_buf, @selector(commit));
  } else {
    [cmd_buf commit];
  }

  id<MTLCommandBuffer> pending_cmd_buf = cmd_buf;
  if (mps_cmd_buf && [mps_cmd_buf respondsToSelector:@selector(commandBuffer)]) {
    id (*msg_send_obj)(id, SEL) = (id (*)(id, SEL))objc_msgSend;
    id raw = msg_send_obj(mps_cmd_buf, @selector(commandBuffer));
    if (raw) {
      pending_cmd_buf = (id<MTLCommandBuffer>)raw;
    }
  }

  for (size_t i = 0; i < g_tls_pending_buffer_count; ++i) {
    StBuffer *buf = g_tls_pending_buffers[i];
    if (!buf) {
      continue;
    }
    st_buffer_track_pending_cmd(buf, (__bridge_retained void *)pending_cmd_buf);
  }

  if (out_pending_cmd_buf) {
    *out_pending_cmd_buf = pending_cmd_buf;
  }

  st_mps_stream_tls_reset_state();
  return true;
}

bool st_mps_stream_finalize_encoded_command_buffer(
    id<MTLCommandBuffer> cmd_buf, id mps_cmd_buf, bool force_commit,
    id<MTLCommandBuffer> *out_pending_cmd_buf) {
  if (!cmd_buf) {
    return false;
  }

  if (out_pending_cmd_buf) {
    *out_pending_cmd_buf = nil;
  }

  st_mps_stream_init_commit_every_once();
  const size_t commit_every =
      atomic_load_explicit(&g_stream_commit_every, memory_order_acquire);

  if (commit_every > 1u) {
    if (st_mps_stream_tls_cmd() != cmd_buf) {
      st_mps_stream_tls_set_cmd(cmd_buf);
    }
    if (mps_cmd_buf && st_mps_stream_tls_mps_cmd() != mps_cmd_buf) {
      st_mps_stream_tls_set_mps_cmd(mps_cmd_buf);
    }
  }

  g_tls_encode_count += 1u;
  const bool should_commit =
      force_commit || commit_every <= 1u || g_tls_encode_count >= commit_every;
  if (!should_commit) {
    return true;
  }

  id<MTLCommandBuffer> commit_cmd = st_mps_stream_tls_cmd();
  if (!commit_cmd) {
    commit_cmd = cmd_buf;
  }
  id commit_mps = st_mps_stream_tls_mps_cmd();
  if (!commit_mps) {
    commit_mps = mps_cmd_buf;
  }

  return st_mps_stream_commit_internal(commit_cmd, commit_mps,
                                       out_pending_cmd_buf);
}

void st_mps_stream_flush(void) {
  id<MTLCommandBuffer> cmd_buf = st_mps_stream_tls_cmd();
  if (!cmd_buf) {
    return;
  }
  id mps_cmd_buf = st_mps_stream_tls_mps_cmd();
  (void)st_mps_stream_commit_internal(cmd_buf, mps_cmd_buf, NULL);
}
