#include "st.h"
#include "st_backend.h"
#include "st_conv.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_BATCH_OPS 8

typedef struct ProfileSums {
  size_t samples;
  double gpu_ms;
  StBufferGpuProfile host;
} ProfileSums;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double ns_to_ms(uint64_t ns) {
  return (double)ns / 1000000.0;
}

static FloatTensor *make4d(size_t n, size_t c, size_t h, size_t w) {
  size_t shape[4] = {n, c, h, w};
  return st_create(4, shape);
}

static void fill_rand(FloatTensor *t, uint32_t seed) {
  for (size_t i = 0; i < t->numel; ++i) {
    seed = seed * 1664525u + 1013904223u;
    t->values[i] = ((float)(seed >> 8) / 16777216.0f) - 0.5f;
  }
}

static float max_abs_diff(const FloatTensor *a, const FloatTensor *b) {
  float max_abs = 0.0f;
  if (!a || !b || a->numel != b->numel) return -1.0f;
  for (size_t i = 0; i < a->numel; ++i) {
    float d = fabsf(a->values[i] - b->values[i]);
    if (d > max_abs) max_abs = d;
  }
  return max_abs;
}

static void collect_profile(ProfileSums *sums, const FloatTensor *out) {
  if (!sums || !out || !out->buf) return;

  double gpu_ms = 0.0;
  StBufferGpuProfile host = {0};
  const bool have_gpu = st_buffer_last_gpu_elapsed_ms(out->buf, &gpu_ms);
  const bool have_host = st_buffer_last_gpu_profile(out->buf, &host);
  if (!have_gpu && !have_host) return;

  if (have_gpu) {
    sums->gpu_ms += gpu_ms;
  }
  if (have_host) {
    sums->host.feed_ms += host.feed_ms;
    sums->host.command_ms += host.command_ms;
    sums->host.encode_ms += host.encode_ms;
    sums->host.commit_ms += host.commit_ms;
    sums->host.sync_wait_ms += host.sync_wait_ms;
    sums->host.sync_wait_prewrite_ms += host.sync_wait_prewrite_ms;
    sums->host.sync_wait_boundary_ms += host.sync_wait_boundary_ms;
  }
  sums->samples++;
}

static void print_header(void) {
  printf("suite,case_name,mode,batch_ops,repeats,ops,total_ms_per_repeat,"
         "ms_per_op,enqueue_ms_per_op,sync_ms_per_op,gpu_avg_ms,"
         "cpu_overhead_ms,profile_samples,feed_avg_ms,command_avg_ms,"
         "encode_avg_ms,commit_avg_ms,sync_wait_avg_ms,"
         "sync_wait_prewrite_avg_ms,sync_wait_boundary_avg_ms,mps_hit,mps_miss,"
         "fallback_gemm,fallback_ref,readbytes_delta,fastpath_delta,"
         "max_abs_diff\n");
}

static void print_ms_or_na(bool valid, double value) {
  if (valid) {
    printf("%.6f", value);
  } else {
    printf("na");
  }
}

static void print_row(const char *mode, size_t batch_ops, size_t repeats,
                      double total_ms, double enqueue_ms, double sync_ms,
                      const ProfileSums *profiles, StBackendCounters before,
                      StBackendCounters after, float max_diff) {
  const size_t ops = batch_ops * repeats;
  const double total_ms_per_repeat = total_ms / (double)repeats;
  const double ms_per_op = total_ms / (double)ops;
  const double enqueue_ms_per_op = enqueue_ms / (double)ops;
  const double sync_ms_per_op = sync_ms / (double)ops;
  const bool have_profile = profiles && profiles->samples > 0;
  const double samples = have_profile ? (double)profiles->samples : 1.0;
  const double gpu_avg = have_profile ? profiles->gpu_ms / samples : 0.0;
  const double cpu_overhead = have_profile ? ms_per_op - gpu_avg : 0.0;

  printf("conv_medium_batch_profile,conv_medium,%s,%zu,%zu,%zu,%.6f,%.6f,"
         "%.6f,%.6f,",
         mode, batch_ops, repeats, ops, total_ms_per_repeat, ms_per_op,
         enqueue_ms_per_op, sync_ms_per_op);
  print_ms_or_na(have_profile, gpu_avg);
  printf(",");
  print_ms_or_na(have_profile, cpu_overhead);
  printf(",%zu,", have_profile ? profiles->samples : 0);
  print_ms_or_na(have_profile, profiles->host.feed_ms / samples);
  printf(",");
  print_ms_or_na(have_profile, profiles->host.command_ms / samples);
  printf(",");
  print_ms_or_na(have_profile, profiles->host.encode_ms / samples);
  printf(",");
  print_ms_or_na(have_profile, profiles->host.commit_ms / samples);
  printf(",");
  print_ms_or_na(have_profile, profiles->host.sync_wait_ms / samples);
  printf(",");
  print_ms_or_na(have_profile,
                 profiles->host.sync_wait_prewrite_ms / samples);
  printf(",");
  print_ms_or_na(have_profile,
                 profiles->host.sync_wait_boundary_ms / samples);
  printf(",%ld,%ld,%ld,%ld,%ld,%ld,",
         after.mps_hit - before.mps_hit,
         after.mps_miss - before.mps_miss,
         after.fallback_gemm - before.fallback_gemm,
         after.fallback_ref - before.fallback_ref,
         after.conv_readbytes - before.conv_readbytes,
         after.conv_fastpath_hit - before.conv_fastpath_hit);
  if (max_diff >= 0.0f) {
    printf("%.8f\n", (double)max_diff);
  } else {
    printf("na\n");
  }
}

static int run_serial_sync_each(const StConv2dParams *params,
                                const FloatTensor *input,
                                const FloatTensor *weight,
                                const FloatTensor *ref,
                                FloatTensor **outputs,
                                size_t batch_ops,
                                size_t repeats) {
  ProfileSums profiles = {0};
  double enqueue_ms = 0.0;
  double sync_ms = 0.0;
  float max_diff = 0.0f;

  StBackendCounters before = st_backend_get_counters();
  const uint64_t total_t0 = now_ns();
  for (size_t r = 0; r < repeats; ++r) {
    for (size_t i = 0; i < batch_ops; ++i) {
      const uint64_t enq_t0 = now_ns();
      if (!st_conv2d_nchw(input, weight, NULL, params, outputs[0])) return 1;
      const uint64_t enq_t1 = now_ns();
      st_tensor_sync(outputs[0]);
      const uint64_t sync_t1 = now_ns();
      enqueue_ms += ns_to_ms(enq_t1 - enq_t0);
      sync_ms += ns_to_ms(sync_t1 - enq_t1);
      collect_profile(&profiles, outputs[0]);
    }
  }
  const uint64_t total_t1 = now_ns();
  StBackendCounters after = st_backend_get_counters();

  max_diff = max_abs_diff(ref, outputs[0]);
  print_row("serial_sync_each", batch_ops, repeats, ns_to_ms(total_t1 - total_t0),
            enqueue_ms, sync_ms, &profiles, before, after, max_diff);
  return 0;
}

static int run_batched_sync_end(const StConv2dParams *params,
                                const FloatTensor *input,
                                const FloatTensor *weight,
                                const FloatTensor *ref,
                                FloatTensor **outputs,
                                size_t batch_ops,
                                size_t repeats) {
  ProfileSums profiles = {0};
  double enqueue_ms = 0.0;
  double sync_ms = 0.0;
  float max_diff = 0.0f;

  StBackendCounters before = st_backend_get_counters();
  const uint64_t total_t0 = now_ns();
  for (size_t r = 0; r < repeats; ++r) {
    const uint64_t enq_t0 = now_ns();
    for (size_t i = 0; i < batch_ops; ++i) {
      if (!st_conv2d_nchw(input, weight, NULL, params, outputs[i])) return 1;
    }
    const uint64_t enq_t1 = now_ns();
    for (size_t i = 0; i < batch_ops; ++i) {
      st_tensor_sync(outputs[i]);
      collect_profile(&profiles, outputs[i]);
    }
    const uint64_t sync_t1 = now_ns();
    enqueue_ms += ns_to_ms(enq_t1 - enq_t0);
    sync_ms += ns_to_ms(sync_t1 - enq_t1);
  }
  const uint64_t total_t1 = now_ns();
  StBackendCounters after = st_backend_get_counters();

  for (size_t i = 0; i < batch_ops; ++i) {
    const float diff = max_abs_diff(ref, outputs[i]);
    if (diff > max_diff) max_diff = diff;
  }
  print_row("batched_sync_end", batch_ops, repeats, ns_to_ms(total_t1 - total_t0),
            enqueue_ms, sync_ms, &profiles, before, after, max_diff);
  return 0;
}

int main(void) {
  int rc = 1;
  const size_t n = 4, c_in = 32, c_out = 64, h = 56, w = 56, k = 3;
  const size_t stride = 1, pad = 1, repeats = 10;
  const size_t batch_sizes[] = {1, 2, 4, 8};
  size_t out_h = 0, out_w = 0;

  StConv2dParams params = st_conv2d_default_params();
  params.backend = ST_CONV_BACKEND_MPS;
  params.stride_h = stride;
  params.stride_w = stride;
  params.pad_h = pad;
  params.pad_w = pad;

  if (!st_conv2d_output_hw(h, w, k, k, &params, &out_h, &out_w)) return 1;

  FloatTensor *input = make4d(n, c_in, h, w);
  FloatTensor *weight = make4d(c_out, c_in, k, k);
  FloatTensor *ref = make4d(n, c_out, out_h, out_w);
  FloatTensor *outputs[MAX_BATCH_OPS] = {0};
  if (!input || !weight || !ref) goto cleanup;
  for (size_t i = 0; i < MAX_BATCH_OPS; ++i) {
    outputs[i] = make4d(n, c_out, out_h, out_w);
    if (!outputs[i]) goto cleanup;
  }

  fill_rand(input, 1u);
  fill_rand(weight, 2u);

  StConv2dParams ref_params = params;
  ref_params.backend = ST_CONV_BACKEND_GEMM;
  if (!st_conv2d_nchw(input, weight, NULL, &ref_params, ref)) goto cleanup;

  st_backend_set_conv_mps_async(true);
  if (!st_conv2d_nchw(input, weight, NULL, &params, outputs[0])) goto cleanup;
  st_tensor_sync(outputs[0]);

  print_header();
  for (size_t i = 0; i < sizeof(batch_sizes) / sizeof(batch_sizes[0]); ++i) {
    const size_t batch_ops = batch_sizes[i];
    st_backend_reset_counters();
    if (run_serial_sync_each(&params, input, weight, ref, outputs,
                             batch_ops, repeats) != 0) {
      goto cleanup;
    }
    st_backend_reset_counters();
    if (run_batched_sync_end(&params, input, weight, ref, outputs,
                             batch_ops, repeats) != 0) {
      goto cleanup;
    }
  }

  rc = 0;

cleanup:
  st_backend_set_conv_mps_async(false);
  for (size_t i = 0; i < MAX_BATCH_OPS; ++i) {
    st_destroy(outputs[i]);
  }
  st_destroy(ref);
  st_destroy(weight);
  st_destroy(input);
  return rc;
}
