/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * st_backend_mps.m — MPS backend vtable for tensor operations
 *
 * Implements conv2d with zero-copy Metal buffer support.
 * Pool and batchnorm forward delegate to the existing st_mps.h functions.
 */

#import "st_backend.h"
#import "st_buffer.h"
#import "st_buffer_metal.h"
#import "st_conv.h"
#import "st_mps.h"
#import "st_pool.h"
#import "sm_mps.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <objc/message.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Shared device / queue (reuse from sm_mps) ---- */

static id<MTLDevice> _st_be_mps_device(void) {
  return (__bridge id<MTLDevice>)mps_get_shared_device();
}

static id<MTLCommandQueue> _st_be_mps_queue(void) {
  return (__bridge id<MTLCommandQueue>)mps_get_shared_command_queue();
}

static double st_backend_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/* ================================================================== */
/*  Thresholds                                                         */
/* ================================================================== */

#define ST_MPS_POOL_THRESHOLD_DEFAULT 4096u
#define ST_MPS_BN_THRESHOLD_DEFAULT   4096u
#define ST_MPS_CONV_ASYNC_DEFAULT     false
#define ST_MPS_CONV_NHWC_DEFAULT      false

static size_t g_mps_pool_threshold = ST_MPS_POOL_THRESHOLD_DEFAULT;
static size_t g_mps_bn_threshold = ST_MPS_BN_THRESHOLD_DEFAULT;
static atomic_bool g_mps_thresholds_initialized = false;
static atomic_bool g_mps_conv_async_enabled = ST_MPS_CONV_ASYNC_DEFAULT;
static atomic_bool g_mps_conv_async_initialized = false;
static atomic_bool g_mps_conv_nhwc_enabled = ST_MPS_CONV_NHWC_DEFAULT;
static atomic_bool g_mps_conv_nhwc_initialized = false;

/* Conv thresholds are read from st_conv via the public threshold API. */

static bool st_backend_parse_positive_size_t(const char *text,
                                             size_t *out_value) {
  if (text == NULL || out_value == NULL) {
    return false;
  }

  errno = 0;
  char *end = NULL;
  const unsigned long long value = strtoull(text, &end, 10);
  if (errno != 0 || end == text || (end != NULL && *end != '\0')) {
    return false;
  }
  if (value == 0ull || value > (unsigned long long)SIZE_MAX) {
    return false;
  }

  *out_value = (size_t)value;
  return true;
}

static bool st_backend_parse_bool(const char *text, bool *out_value) {
  if (text == NULL || out_value == NULL) {
    return false;
  }

  if (strcmp(text, "1") == 0 || strcmp(text, "true") == 0 ||
      strcmp(text, "TRUE") == 0 || strcmp(text, "yes") == 0 ||
      strcmp(text, "YES") == 0 || strcmp(text, "on") == 0 ||
      strcmp(text, "ON") == 0) {
    *out_value = true;
    return true;
  }

  if (strcmp(text, "0") == 0 || strcmp(text, "false") == 0 ||
      strcmp(text, "FALSE") == 0 || strcmp(text, "no") == 0 ||
      strcmp(text, "NO") == 0 || strcmp(text, "off") == 0 ||
      strcmp(text, "OFF") == 0) {
    *out_value = false;
    return true;
  }

  return false;
}

static void st_backend_init_mps_thresholds_once(void) {
  if (atomic_load_explicit(&g_mps_thresholds_initialized,
                           memory_order_acquire)) {
    return;
  }
  st_backend_reload_mps_thresholds_from_env();
}

static void st_backend_init_conv_async_once(void) {
  if (atomic_load_explicit(&g_mps_conv_async_initialized,
                           memory_order_acquire)) {
    return;
  }
  st_backend_reload_conv_mps_async_from_env();
}

static void st_backend_init_conv_nhwc_once(void) {
  if (atomic_load_explicit(&g_mps_conv_nhwc_initialized,
                           memory_order_acquire)) {
    return;
  }
  st_backend_reload_conv_mps_nhwc_from_env();
}

/* ================================================================== */
/*  MPSGraph cache (NSCache — thread-safe, auto-evict LRU)             */
/* ================================================================== */

static NSCache<NSString *, NSDictionary *> *_st_mps_graph_cache(void) {
  static NSCache<NSString *, NSDictionary *> *cache = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    cache = [[NSCache alloc] init];
    cache.countLimit = 16;
  });
  return cache;
}

static bool st_mps_tensor_is_valid(const FloatTensor *t, size_t ndim) {
  return t != NULL && t->values != NULL && t->ndim == ndim;
}

/* ================================================================== */
/*  supports_op                                                        */
/* ================================================================== */

static bool mps_supports_op(StOp op, size_t numel,
                            StBufferType buf_type __attribute__((unused))) {
  st_backend_init_mps_thresholds_once();

  switch (op) {
    case ST_OP_CONV2D_FORWARD: {
      /* Conv threshold logic is handled inside st_conv.c (AUTO mode)
       * and inside conv2d_forward below.  Always report "supported"
       * so the caller can attempt MPS; the function returns false
       * when thresholds aren't met. */
      return true;
    }
    case ST_OP_MAXPOOL2D_FORWARD:
    case ST_OP_AVGPOOL2D_FORWARD:
      return numel >= g_mps_pool_threshold;
    case ST_OP_BATCHNORM2D_FORWARD:
      return numel >= g_mps_bn_threshold;
    default:
      return false;
  }
}

/* ================================================================== */
/*  Conv2D forward — MPSGraph with zero-copy input                     */
/* ================================================================== */

static bool mps_conv2d_forward(const FloatTensor *input,
                               const FloatTensor *weight,
                               const FloatTensor *bias,
                               const StConv2dParams *params,
                               FloatTensor *output) {
  if (!input || !weight || !output || !params) return false;

  const size_t n     = input->shape[0];
  const size_t c_in  = input->shape[1];
  const size_t h_in  = input->shape[2];
  const size_t w_in  = input->shape[3];
  const size_t c_out = weight->shape[0];
  const size_t k_h   = weight->shape[2];
  const size_t k_w   = weight->shape[3];
  @autoreleasepool {

  id<MTLDevice> device = _st_be_mps_device();
  id<MTLCommandQueue> queue = _st_be_mps_queue();
  if (!device || !queue) return false;

  st_backend_init_conv_async_once();
  st_backend_init_conv_nhwc_once();
  const bool use_nhwc = atomic_load_explicit(&g_mps_conv_nhwc_enabled,
                                             memory_order_acquire);

  /* ---- Cache lookup by shape signature ---- */
  const int has_bias = (bias != NULL) ? 1 : 0;
  NSString *cacheKey = [NSString stringWithFormat:
      @"conv:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,nhwc=%d",
      n, c_in, h_in, w_in, c_out, k_h, k_w,
      params->stride_h, params->stride_w,
      params->pad_h, params->pad_w,
      params->dilation_h, params->dilation_w, has_bias, (int)use_nhwc];

  NSCache *cache = _st_mps_graph_cache();
  NSDictionary *cached = [cache objectForKey:cacheKey];

  MPSGraph *graph;
  MPSGraphExecutable *executable = nil;
  MPSGraphTensor *inT, *wT, *biasT, *resultT;

  if (cached) {
    /* Cache hit — reuse compiled graph. */
    graph      = cached[@"graph"];
    executable = cached[@"executable"];
    if ([executable isEqual:[NSNull null]]) executable = nil;
    inT        = cached[@"inT"];
    wT         = cached[@"wT"];
    biasT      = cached[@"biasT"];   /* may be NSNull */
    resultT    = cached[@"resultT"];
    if ([biasT isEqual:[NSNull null]]) biasT = nil;
  } else {
    /* Cache miss — build graph. */
    graph = [[MPSGraph alloc] init];

    MPSShape *inShape = @[ @(n), @(c_in), @(h_in), @(w_in) ];
    MPSShape *wShape  = @[ @(c_out), @(c_in), @(k_h), @(k_w) ];

    inT = [graph placeholderWithShape:inShape
                             dataType:MPSDataTypeFloat32
                                 name:@"input"];
    wT  = [graph placeholderWithShape:wShape
                             dataType:MPSDataTypeFloat32
                                 name:@"weight"];

    /* Optional NCHW→NHWC transpose inside the graph (no host copy). */
    MPSGraphTensor *convSrcT;
    if (use_nhwc) {
      MPSGraphTensor *t1 = [graph transposeTensor:inT
                                        dimension:1 withDimension:2
                                             name:@"nchw2nhwc_1"];
      convSrcT = [graph transposeTensor:t1
                              dimension:2 withDimension:3
                                   name:@"nchw2nhwc_2"];
    } else {
      convSrcT = inT;
    }

    MPSGraphConvolution2DOpDescriptor *convDesc =
        [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:(NSUInteger)params->stride_w
                          strideInY:(NSUInteger)params->stride_h
                    dilationRateInX:(NSUInteger)params->dilation_w
                    dilationRateInY:(NSUInteger)params->dilation_h
                             groups:1
                        paddingLeft:(NSUInteger)params->pad_w
                       paddingRight:(NSUInteger)params->pad_w
                         paddingTop:(NSUInteger)params->pad_h
                      paddingBottom:(NSUInteger)params->pad_h
                       paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:(use_nhwc ? MPSGraphTensorNamedDataLayoutNHWC
                                             : MPSGraphTensorNamedDataLayoutNCHW)
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    MPSGraphTensor *convT = [graph convolution2DWithSourceTensor:convSrcT
                                                   weightsTensor:wT
                                                      descriptor:convDesc
                                                            name:@"conv2d"];

    /* Optional NHWC→NCHW transpose back after conv. */
    MPSGraphTensor *convOutT;
    if (use_nhwc) {
      /* conv output is [N, H_out, W_out, C_out] (NHWC) — transpose back */
      MPSGraphTensor *t1 = [graph transposeTensor:convT
                                        dimension:2 withDimension:3
                                             name:@"nhwc2nchw_1"];
      convOutT = [graph transposeTensor:t1
                              dimension:1 withDimension:2
                                   name:@"nhwc2nchw_2"];
    } else {
      convOutT = convT;
    }

    resultT = convOutT;
    biasT   = nil;

    if (bias) {
      biasT   = [graph placeholderWithShape:@[ @1, @(c_out), @1, @1 ]
                                   dataType:MPSDataTypeFloat32
                                       name:@"bias"];
      resultT = [graph additionWithPrimaryTensor:convOutT
                                 secondaryTensor:biasT
                                            name:@"add_bias"];
    }

    /* Compile executable for async inference path. */
    NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feedShapes =
        [NSMutableDictionary dictionaryWithCapacity:3];
    feedShapes[inT] = [[MPSGraphShapedType alloc]
        initWithShape:inShape dataType:MPSDataTypeFloat32];
    feedShapes[wT] = [[MPSGraphShapedType alloc]
        initWithShape:wShape dataType:MPSDataTypeFloat32];
    if (biasT) {
      feedShapes[biasT] = [[MPSGraphShapedType alloc]
          initWithShape:@[ @1, @(c_out), @1, @1 ]
                dataType:MPSDataTypeFloat32];
    }
    executable = [graph
        compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                    feeds:feedShapes
            targetTensors:@[ resultT ]
         targetOperations:nil
     compilationDescriptor:nil];

    /* Store into cache. */
    [cache setObject:@{
      @"graph"      : graph,
      @"executable" : executable ?: [NSNull null],
      @"inT"        : inT,
      @"wT"         : wT,
      @"biasT"      : biasT   ?: [NSNull null],
      @"resultT"    : resultT,
    } forKey:cacheKey];
  }

  /* ---- Prepare feed data ---- */
  const double feed_start_ms = st_backend_now_ms();
  MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];

  void *in_mh  = input->buf  ? st_buffer_metal_handle(input->buf)  : NULL;
  void *w_mh   = weight->buf ? st_buffer_metal_handle(weight->buf) : NULL;

  MPSShape *inShape = @[ @(n), @(c_in), @(h_in), @(w_in) ];
  MPSShape *wShape  = @[ @(c_out), @(c_in), @(k_h), @(k_w) ];

  const size_t inBytes = n * c_in * h_in * w_in * sizeof(float);
  const size_t wBytes  = c_out * c_in * k_h * k_w * sizeof(float);

  MPSGraphTensorData *inData = st_mps_make_tensor_data(gDev, input->values,
                                                       in_mh, inBytes, inShape);
  MPSGraphTensorData *wData  = st_mps_make_tensor_data(
      gDev, weight->values, w_mh, wBytes, wShape);

  NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
      [NSMutableDictionary dictionaryWithCapacity:3];
  feeds[inT] = inData;
  feeds[wT]  = wData;

  if (bias && biasT) {
    void *b_mh = bias->buf ? st_buffer_metal_handle(bias->buf) : NULL;
    const size_t bBytes = c_out * sizeof(float);
    MPSGraphTensorData *bData = st_mps_make_tensor_data(
        gDev, bias->values, b_mh, bBytes, @[ @1, @(c_out), @1, @1 ]);
    feeds[biasT] = bData;
  }

  const bool wants_fastpath =
      (output->buf && st_buffer_metal_handle(output->buf));
  if (wants_fastpath && !executable) {
    st_backend_counter_conv_fastpath_executable_nil();
  }

  /* ---- Fastpath: executable + pre-allocated output MTLBuffer ---- */
  if (executable && wants_fastpath) {
    bool fastpath_ok = true;
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedMap =
        [NSMutableDictionary dictionaryWithDictionary:feeds];
    NSMutableArray<MPSGraphTensorData *> *inputsArray =
        [NSMutableArray arrayWithCapacity:executable.feedTensors.count];
    for (MPSGraphTensor *ft in executable.feedTensors) {
      MPSGraphTensorData *td = feedMap[ft];
      if (!td) {
        fastpath_ok = false;
        st_backend_counter_conv_fastpath_missing_feed();
        break;
      }
      [inputsArray addObject:td];
    }

    if (fastpath_ok) {
      MPSShape *outShape = @[ @(output->shape[0]), @(output->shape[1]),
                              @(output->shape[2]), @(output->shape[3]) ];
      id<MTLBuffer> outBuf =
          (__bridge id<MTLBuffer>)st_buffer_metal_handle(output->buf);
      MPSGraphTensorData *preOutData =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                  shape:outShape
                                               dataType:MPSDataTypeFloat32];
      const double feed_end_ms = st_backend_now_ms();
      const double command_start_ms = feed_end_ms;
      id<MTLCommandBuffer> cmdBuf = nil;
      id mpsCmdBuf = nil;
      if ([queue respondsToSelector:@selector(commandBufferWithDescriptor:)]) {
        MTLCommandBufferDescriptor *cbDesc = [[MTLCommandBufferDescriptor alloc] init];
        cmdBuf = [queue commandBufferWithDescriptor:cbDesc];
      }
      if (!cmdBuf) {
        cmdBuf = [queue commandBuffer];
      }

      /* On some runtimes MPSGraphExecutable encode expects an MPSCommandBuffer
         wrapper (selector mpsCommandBufferDescriptor). Create it dynamically
         when available, but keep fallback to plain MTLCommandBuffer. */
      Class MPSCBClass = NSClassFromString(@"MPSCommandBuffer");
      if (MPSCBClass && [MPSCBClass respondsToSelector:@selector(commandBufferFromCommandQueue:)]) {
        id (*msgSendCB)(id, SEL, id) = (id (*)(id, SEL, id))objc_msgSend;
        mpsCmdBuf = msgSendCB(MPSCBClass, @selector(commandBufferFromCommandQueue:), queue);
      }
      const double command_end_ms = st_backend_now_ms();

      if (!preOutData) {
        st_backend_counter_conv_fastpath_preout_nil();
        fastpath_ok = false;
      }
      if (!cmdBuf) {
        st_backend_counter_conv_fastpath_cmd_buf_nil();
        fastpath_ok = false;
      }
      if (fastpath_ok) {
        output->buf->_last_gpu_profile_valid = false;
        output->buf->_last_gpu_profile = (StBufferGpuProfile){0};
        output->buf->_last_gpu_profile.feed_ms = feed_end_ms - feed_start_ms;
        output->buf->_last_gpu_profile.command_ms =
            command_end_ms - command_start_ms;

        /* Standalone async-safe rule: before writing to the same output
           buffer again, drain the previous pending command buffer. */
        if (output->buf->_async_cmd_buf) {
          st_buffer_metal_wait(output->buf);
        }

        const double encode_start_ms = st_backend_now_ms();
        @try {
          id encodeCmdBuf = mpsCmdBuf ? mpsCmdBuf : (id)cmdBuf;
          [executable encodeToCommandBuffer:encodeCmdBuf
                                 inputsArray:inputsArray
                                resultsArray:@[ preOutData ]
                         executionDescriptor:nil];
        } @catch (NSException *exception) {
          st_backend_counter_conv_fastpath_encode_exception();
          NSLog(@"mps_conv2d fastpath encode exception: %@ (%@)",
                exception.name, exception.reason);
          fastpath_ok = false;
        }
        const double encode_end_ms = st_backend_now_ms();
        output->buf->_last_gpu_profile.encode_ms =
            encode_end_ms - encode_start_ms;
        if (fastpath_ok) {
          const double commit_start_ms = st_backend_now_ms();
          if (mpsCmdBuf && [mpsCmdBuf respondsToSelector:@selector(commit)]) {
            void (*msgSendVoid)(id, SEL) = (void (*)(id, SEL))objc_msgSend;
            msgSendVoid(mpsCmdBuf, @selector(commit));
          } else {
            [cmdBuf commit];
          }

          id<MTLCommandBuffer> pendingCmdBuf = cmdBuf;
          if (mpsCmdBuf && [mpsCmdBuf respondsToSelector:@selector(commandBuffer)]) {
            id (*msgSendCBObj)(id, SEL) = (id (*)(id, SEL))objc_msgSend;
            id raw = msgSendCBObj(mpsCmdBuf, @selector(commandBuffer));
            if (raw) pendingCmdBuf = (id<MTLCommandBuffer>)raw;
          }
          const double commit_end_ms = st_backend_now_ms();
          output->buf->_last_gpu_profile.commit_ms =
              commit_end_ms - commit_start_ms;
          output->buf->_last_gpu_profile_valid = true;

          if (st_backend_get_conv_mps_async()) {
            output->buf->_async_cmd_buf = (__bridge_retained void *)pendingCmdBuf;
          } else {
            [pendingCmdBuf waitUntilCompleted];
          }
          st_backend_counter_conv_fastpath_hit();
          return true;
        }
      }
    }
  }

  /* ---- Fallback: run graph synchronously + readback ---- */
  NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = nil;

  @try {
    results = [graph runWithMTLCommandQueue:queue
                                     feeds:feeds
                             targetTensors:@[ resultT ]
                          targetOperations:nil];
  } @catch (NSException *exception) {
    return false;
  }

  MPSGraphTensorData *outData = results[resultT];
  if (!outData) return false;

  st_backend_counter_conv_readbytes();
  [outData.mpsndarray readBytes:output->values strideBytes:nil];
  return true;

  } // @autoreleasepool
}

/* ================================================================== */
/*  Pool forward — delegate to existing st_mps.h functions             */
/* ================================================================== */

static bool mps_maxpool2d_forward(const FloatTensor *input, size_t kh,
                                  size_t kw, size_t sh, size_t sw, size_t ph,
                                  size_t pw, FloatTensor *output,
                                  FloatTensor *indices) {
  if (indices) return false;  /* MPS path doesn't produce indices */

  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];
  const size_t oh = output->shape[2];
  const size_t ow = output->shape[3];

  void *mh = input->buf ? st_buffer_metal_handle(input->buf) : NULL;
  return st_maxpool2d_mps(input->values, mh, n, c, h, w, kh, kw,
                          sh, sw, ph, pw, output->values, oh, ow);
}

static bool mps_avgpool2d_forward(const FloatTensor *input, size_t kh,
                                  size_t kw, size_t sh, size_t sw, size_t ph,
                                  size_t pw, FloatTensor *output) {
  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];
  const size_t oh = output->shape[2];
  const size_t ow = output->shape[3];

  void *mh = input->buf ? st_buffer_metal_handle(input->buf) : NULL;
  return st_avgpool2d_mps(input->values, mh, n, c, h, w, kh, kw,
                          sh, sw, ph, pw, output->values, oh, ow);
}

/* ================================================================== */
/*  Batchnorm forward — delegate to existing st_mps.h function        */
/* ================================================================== */

static bool mps_batchnorm2d_forward(const FloatTensor *input,
                                    const FloatTensor *gamma,
                                    const FloatTensor *beta, float epsilon,
                                    FloatTensor *output, FloatTensor *mean,
                                    FloatTensor *var) {
  const size_t n = input->shape[0];
  const size_t c = input->shape[1];
  const size_t h = input->shape[2];
  const size_t w = input->shape[3];

  void *mh = input->buf ? st_buffer_metal_handle(input->buf) : NULL;
  return st_batchnorm2d_forward_mps(
      input->values, mh, n, c, h, w,
      gamma ? gamma->values : NULL, beta ? beta->values : NULL,
      epsilon, output->values,
      mean ? mean->values : NULL, var ? var->values : NULL);
}

bool st_backend_conv2d_batchnorm2d_forward_mps(
    const FloatTensor *input, const FloatTensor *weight,
    const FloatTensor *bias, const StConv2dParams *params,
    const FloatTensor *gamma, const FloatTensor *beta, float epsilon,
    FloatTensor *output, FloatTensor *mean, FloatTensor *var,
    bool apply_relu) {
  if (!input || !weight || !params || !output) return false;
  if (params->backend != ST_CONV_BACKEND_AUTO &&
      params->backend != ST_CONV_BACKEND_MPS) {
    return false;
  }

  if (input->dtype != ST_DTYPE_F32 || weight->dtype != ST_DTYPE_F32 ||
      output->dtype != ST_DTYPE_F32 ||
      (mean && mean->dtype != ST_DTYPE_F32) ||
      (var  && var->dtype  != ST_DTYPE_F32)) {
    return false;
  }
  if ((bias && bias->dtype != ST_DTYPE_F32) ||
      (gamma && gamma->dtype != ST_DTYPE_F32) ||
      (beta && beta->dtype != ST_DTYPE_F32)) {
    return false;
  }

  if (!st_mps_tensor_is_valid(input, 4) ||
      !st_mps_tensor_is_valid(weight, 4) ||
      !st_mps_tensor_is_valid(output, 4) ||
      (mean && !st_mps_tensor_is_valid(mean, 1)) ||
      (var  && !st_mps_tensor_is_valid(var,  1))) {
    return false;
  }
  if (!st_is_contiguous(input) || !st_is_contiguous(weight) ||
      !st_is_contiguous(output) ||
      (mean && !st_is_contiguous(mean)) ||
      (var  && !st_is_contiguous(var))) {
    return false;
  }
  if (bias && (!st_mps_tensor_is_valid(bias, 1) || !st_is_contiguous(bias))) {
    return false;
  }
  if (gamma &&
      (!st_mps_tensor_is_valid(gamma, 1) || !st_is_contiguous(gamma))) {
    return false;
  }
  if (beta && (!st_mps_tensor_is_valid(beta, 1) || !st_is_contiguous(beta))) {
    return false;
  }

  const size_t n = input->shape[0];
  const size_t c_in = input->shape[1];
  const size_t h_in = input->shape[2];
  const size_t w_in = input->shape[3];
  const size_t c_out = weight->shape[0];
  const size_t w_c_in = weight->shape[1];
  const size_t k_h = weight->shape[2];
  const size_t k_w = weight->shape[3];

  if (c_in != w_c_in) return false;
  if (bias && bias->shape[0] != c_out) return false;
  if (gamma && gamma->shape[0] != c_out) return false;
  if (beta && beta->shape[0] != c_out) return false;
  if ((mean && mean->shape[0] != c_out) ||
      (var  && var->shape[0]  != c_out)) return false;

  size_t out_h = 0;
  size_t out_w = 0;
  if (!st_conv2d_output_hw(h_in, w_in, k_h, k_w, params, &out_h, &out_w)) {
    return false;
  }
  if (output->shape[0] != n || output->shape[1] != c_out ||
      output->shape[2] != out_h || output->shape[3] != out_w) {
    return false;
  }

  if (params->backend == ST_CONV_BACKEND_AUTO) {
    const StBackend *mps = st_backend_mps();
    const StBackend *dflt = st_get_default_backend();
    if (!mps || (dflt && dflt != mps)) {
      return false;
    }

    double macs_threshold = 0.0;
    size_t out_elems_threshold = 0;
    st_conv_get_mps_thresholds(&macs_threshold, &out_elems_threshold);

    const double macs =
        (double)n * (double)c_out * (double)out_h * (double)out_w *
        (double)c_in * (double)k_h * (double)k_w;
    const size_t out_elems = n * c_out * out_h * out_w;
    if (macs < macs_threshold || out_elems < out_elems_threshold) {
      return false;
    }
  }

  @autoreleasepool {
    id<MTLDevice> device = _st_be_mps_device();
    id<MTLCommandQueue> queue = _st_be_mps_queue();
    if (!device || !queue) return false;

    const int has_bias = (bias != NULL) ? 1 : 0;
    const int has_gamma = (gamma != NULL) ? 1 : 0;
    const int has_beta = (beta != NULL) ? 1 : 0;
    const int do_relu = apply_relu ? 1 : 0;
    const int has_mean_out = (mean != NULL) ? 1 : 0;
    const int has_var_out  = (var  != NULL) ? 1 : 0;
    NSString *cacheKey = [NSString stringWithFormat:
        @"convbn:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,%d,%d,%d,%d,%d,%.9g",
        n, c_in, h_in, w_in, c_out, k_h, k_w,
        params->stride_h, params->stride_w,
        params->pad_h, params->pad_w,
        params->dilation_h, params->dilation_w,
        has_bias, has_gamma, has_beta, do_relu,
        has_mean_out, has_var_out, (double)epsilon];

    NSCache *cache = _st_mps_graph_cache();
    NSDictionary *cached = [cache objectForKey:cacheKey];

    MPSGraph *graph;
    MPSGraphTensor *inT, *wT, *biasT, *gammaT, *betaT;
    MPSGraphTensor *resultT, *meanT, *varT;
    MPSGraphExecutable *executable = nil;

    if (cached) {
      graph = cached[@"graph"];
      executable = cached[@"executable"];
      if ([executable isEqual:[NSNull null]]) executable = nil;
      inT = cached[@"inT"];
      wT = cached[@"wT"];
      biasT = cached[@"biasT"];
      gammaT = cached[@"gammaT"];
      betaT = cached[@"betaT"];
      resultT = cached[@"resultT"];
      meanT = cached[@"meanT"];
      varT = cached[@"varT"];
      if ([biasT isEqual:[NSNull null]]) biasT = nil;
      if ([gammaT isEqual:[NSNull null]]) gammaT = nil;
      if ([betaT isEqual:[NSNull null]]) betaT = nil;
    } else {
      graph = [[MPSGraph alloc] init];

      MPSShape *inShape = @[ @(n), @(c_in), @(h_in), @(w_in) ];
      MPSShape *wShape = @[ @(c_out), @(c_in), @(k_h), @(k_w) ];

      inT = [graph placeholderWithShape:inShape
                               dataType:MPSDataTypeFloat32
                                   name:@"input"];
      wT = [graph placeholderWithShape:wShape
                              dataType:MPSDataTypeFloat32
                                  name:@"weight"];

      MPSGraphConvolution2DOpDescriptor *convDesc =
          [MPSGraphConvolution2DOpDescriptor
              descriptorWithStrideInX:(NSUInteger)params->stride_w
                            strideInY:(NSUInteger)params->stride_h
                      dilationRateInX:(NSUInteger)params->dilation_w
                      dilationRateInY:(NSUInteger)params->dilation_h
                               groups:1
                          paddingLeft:(NSUInteger)params->pad_w
                         paddingRight:(NSUInteger)params->pad_w
                           paddingTop:(NSUInteger)params->pad_h
                        paddingBottom:(NSUInteger)params->pad_h
                         paddingStyle:MPSGraphPaddingStyleExplicit
                           dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                        weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

      MPSGraphTensor *convT = [graph convolution2DWithSourceTensor:inT
                                                     weightsTensor:wT
                                                        descriptor:convDesc
                                                              name:@"conv2d"];

      MPSGraphTensor *bnInputT = convT;
      biasT = nil;
      gammaT = nil;
      betaT = nil;

      if (bias) {
        biasT = [graph placeholderWithShape:@[ @1, @(c_out), @1, @1 ]
                                   dataType:MPSDataTypeFloat32
                                       name:@"bias"];
        bnInputT = [graph additionWithPrimaryTensor:bnInputT
                                    secondaryTensor:biasT
                                               name:@"add_bias"];
      }

      NSArray<NSNumber *> *reduceAxes = @[ @0, @2, @3 ];
      meanT = [graph meanOfTensor:bnInputT axes:reduceAxes name:@"mean"];

      MPSGraphTensor *diffT =
          [graph subtractionWithPrimaryTensor:bnInputT
                              secondaryTensor:meanT
                                         name:@"diff"];
      MPSGraphTensor *sqDiffT =
          [graph squareWithTensor:diffT name:@"sq_diff"];
      varT = [graph meanOfTensor:sqDiffT axes:reduceAxes name:@"var"];

      MPSGraphTensor *epsT =
          [graph constantWithScalar:(double)epsilon
                              shape:@[ @1 ]
                           dataType:MPSDataTypeFloat32];
      MPSGraphTensor *varPlusEpsT =
          [graph additionWithPrimaryTensor:varT
                           secondaryTensor:epsT
                                      name:@"var_eps"];
      MPSGraphTensor *sqrtVarT =
          [graph squareRootWithTensor:varPlusEpsT name:@"sqrt_var"];
      MPSGraphTensor *invStdT =
          [graph reciprocalWithTensor:sqrtVarT name:@"inv_std"];

      resultT = [graph multiplicationWithPrimaryTensor:diffT
                                       secondaryTensor:invStdT
                                                  name:@"normed"];

      if (gamma) {
        gammaT = [graph placeholderWithShape:@[ @1, @(c_out), @1, @1 ]
                                    dataType:MPSDataTypeFloat32
                                        name:@"gamma"];
        resultT = [graph multiplicationWithPrimaryTensor:resultT
                                         secondaryTensor:gammaT
                                                    name:@"scale"];
      }

      if (beta) {
        betaT = [graph placeholderWithShape:@[ @1, @(c_out), @1, @1 ]
                                   dataType:MPSDataTypeFloat32
                                       name:@"beta"];
        resultT = [graph additionWithPrimaryTensor:resultT
                                   secondaryTensor:betaT
                                              name:@"shift"];
      }

      if (apply_relu) {
        resultT = [graph reLUWithTensor:resultT name:@"relu"];
      }

      /* Compile MPSGraphExecutable for inference path (no mean/var readback). */
      if (!has_mean_out && !has_var_out) {
        MPSShape *bnShape = @[ @1, @(c_out), @1, @1 ];
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feedShapes =
            [NSMutableDictionary dictionaryWithCapacity:5];
        feedShapes[inT] = [[MPSGraphShapedType alloc]
            initWithShape:inShape dataType:MPSDataTypeFloat32];
        feedShapes[wT] = [[MPSGraphShapedType alloc]
            initWithShape:wShape dataType:MPSDataTypeFloat32];
        if (biasT)
          feedShapes[biasT] = [[MPSGraphShapedType alloc]
              initWithShape:bnShape dataType:MPSDataTypeFloat32];
        if (gammaT)
          feedShapes[gammaT] = [[MPSGraphShapedType alloc]
              initWithShape:bnShape dataType:MPSDataTypeFloat32];
        if (betaT)
          feedShapes[betaT] = [[MPSGraphShapedType alloc]
              initWithShape:bnShape dataType:MPSDataTypeFloat32];
        executable = [graph
            compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                        feeds:feedShapes
                targetTensors:@[ resultT ]
             targetOperations:nil
         compilationDescriptor:nil];
      }

      [cache setObject:@{
        @"graph"      : graph,
        @"executable" : executable ?: [NSNull null],
        @"inT" : inT,
        @"wT" : wT,
        @"biasT" : biasT ?: [NSNull null],
        @"gammaT" : gammaT ?: [NSNull null],
        @"betaT" : betaT ?: [NSNull null],
        @"resultT" : resultT,
        @"meanT" : meanT,
        @"varT" : varT,
      } forKey:cacheKey];
    }

    MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSShape *inShape = @[ @(n), @(c_in), @(h_in), @(w_in) ];
    MPSShape *wShape = @[ @(c_out), @(c_in), @(k_h), @(k_w) ];
    const size_t inBytes = n * c_in * h_in * w_in * sizeof(float);
    const size_t wBytes = c_out * c_in * k_h * k_w * sizeof(float);

    void *in_mh = input->buf ? st_buffer_metal_handle(input->buf) : NULL;
    void *w_mh = weight->buf ? st_buffer_metal_handle(weight->buf) : NULL;
    MPSGraphTensorData *inData = st_mps_make_tensor_data(
        gDev, input->values, in_mh, inBytes, inShape);
    MPSGraphTensorData *wData = st_mps_make_tensor_data(
        gDev, weight->values, w_mh, wBytes, wShape);

    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
        [NSMutableDictionary dictionaryWithCapacity:5];
    feeds[inT] = inData;
    feeds[wT] = wData;

    if (bias && biasT) {
      void *b_mh = bias->buf ? st_buffer_metal_handle(bias->buf) : NULL;
      feeds[biasT] = st_mps_make_tensor_data(
          gDev, bias->values, b_mh, c_out * sizeof(float),
          @[ @1, @(c_out), @1, @1 ]);
    }

    if (gamma && gammaT) {
      void *g_mh = gamma->buf ? st_buffer_metal_handle(gamma->buf) : NULL;
      feeds[gammaT] = st_mps_make_tensor_data(
          gDev, gamma->values, g_mh, c_out * sizeof(float),
          @[ @1, @(c_out), @1, @1 ]);
    }

    if (beta && betaT) {
      void *b_mh = beta->buf ? st_buffer_metal_handle(beta->buf) : NULL;
      feeds[betaT] = st_mps_make_tensor_data(
          gDev, beta->values, b_mh, c_out * sizeof(float),
          @[ @1, @(c_out), @1, @1 ]);
    }

    /* ---- Inference path: executable + pre-allocated output MTLBuffer ---- */
    if (executable && output->buf && st_buffer_metal_handle(output->buf)) {
      /* Build feedMap for order-independent inputsArray construction. */
      NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedMap =
          [NSMutableDictionary dictionaryWithDictionary:feeds];
      NSMutableArray<MPSGraphTensorData *> *inputsArray =
          [NSMutableArray arrayWithCapacity:executable.feedTensors.count];
      for (MPSGraphTensor *ft in executable.feedTensors) {
        MPSGraphTensorData *td = feedMap[ft];
        if (!td) return false;
        [inputsArray addObject:td];
      }

      /* Pre-allocate result TensorData backed by output's shared MTLBuffer. */
      MPSShape *outShape = @[ @(n), @(c_out), @(out_h), @(out_w) ];
      id<MTLBuffer> outBuf =
          (__bridge id<MTLBuffer>)st_buffer_metal_handle(output->buf);
      MPSGraphTensorData *preOutData =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                  shape:outShape
                                               dataType:MPSDataTypeFloat32];

      /* Async encode: commit without waiting; caller uses st_tensor_sync(). */
      id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
      if (!cmdBuf) return false;
      @try {
        [executable encodeToCommandBuffer:cmdBuf
                               inputsArray:inputsArray
                              resultsArray:@[ preOutData ]
                       executionDescriptor:nil];
      } @catch (NSException *exception) {
        return false;
      }
      [cmdBuf commit];
      /* Drain any previously-pending cmd buffer for this output before
         overwriting (prevents orphaned cmd buffers in iterative loops). */
      if (output->buf->_async_cmd_buf) {
        st_buffer_metal_wait(output->buf);
      }
      /* Store bridge-retained command buffer; st_buffer_metal_wait releases. */
      output->buf->_async_cmd_buf = (__bridge_retained void *)cmdBuf;
      /* Output is already in output->values via shared MTLBuffer — no readBytes. */
      return true;
    }

    /* ---- Training / non-Metal-output path: graph run + readBytes ---- */
    NSMutableArray<MPSGraphTensor *> *targets =
        [NSMutableArray arrayWithObject:resultT];
    if (mean && meanT) [targets addObject:meanT];
    if (var  && varT)  [targets addObject:varT];

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = nil;
    @try {
      results = [graph runWithMTLCommandQueue:queue
                                       feeds:feeds
                               targetTensors:targets
                            targetOperations:nil];
    } @catch (NSException *exception) {
      return false;
    }

    MPSGraphTensorData *outData = results[resultT];
    if (!outData) return false;

    [outData.mpsndarray readBytes:output->values strideBytes:nil];
    if (mean) {
      MPSGraphTensorData *meanData = results[meanT];
      if (meanData) [meanData.mpsndarray readBytes:mean->values strideBytes:nil];
    }
    if (var) {
      MPSGraphTensorData *varData = results[varT];
      if (varData) [varData.mpsndarray readBytes:var->values strideBytes:nil];
    }
    return true;
  }
}

bool st_backend_conv2d_batchnorm2d_pool_forward_mps(
    const FloatTensor *input, const FloatTensor *weight,
    const FloatTensor *bias, const StConv2dParams *conv_params,
    const FloatTensor *gamma, const FloatTensor *beta, float epsilon,
    const StPool2dParams *pool_params,
    FloatTensor *output, FloatTensor *mean, FloatTensor *var,
    bool apply_relu) {

  if (!input || !weight || !conv_params || !pool_params || !output)
    return false;
  if (input->ndim != 4 || weight->ndim != 4 || output->ndim != 4)
    return false;

  const size_t n     = input->shape[0];
  const size_t c_in  = input->shape[1];
  const size_t h_in  = input->shape[2];
  const size_t w_in  = input->shape[3];
  const size_t c_out = weight->shape[0];
  const size_t w_c_in = weight->shape[1];
  const size_t k_h   = weight->shape[2];
  const size_t k_w   = weight->shape[3];

  if (c_in != w_c_in) return false;

  /* Compute conv output size. */
  size_t bn_h = 0, bn_w = 0;
  if (!st_conv2d_output_hw(h_in, w_in, k_h, k_w, conv_params, &bn_h, &bn_w))
    return false;

  /* Compute pool output size. */
  size_t out_h = 0, out_w = 0;
  if (!st_pool2d_output_hw(bn_h, bn_w,
                           pool_params->kernel_h, pool_params->kernel_w,
                           pool_params->stride_h, pool_params->stride_w,
                           pool_params->pad_h, pool_params->pad_w,
                           &out_h, &out_w))
    return false;

  if (output->shape[0] != n || output->shape[1] != c_out ||
      output->shape[2] != out_h || output->shape[3] != out_w)
    return false;

  if ((mean && mean->shape[0] != c_out) ||
      (var  && var->shape[0]  != c_out))
    return false;

  if (conv_params->backend == ST_CONV_BACKEND_AUTO) {
    const StBackend *mps  = st_backend_mps();
    const StBackend *dflt = st_get_default_backend();
    if (!mps || (dflt && dflt != mps)) return false;

    double macs_threshold = 0.0;
    size_t out_elems_threshold = 0;
    st_conv_get_mps_thresholds(&macs_threshold, &out_elems_threshold);
    const double macs =
        (double)n * (double)c_out * (double)bn_h * (double)bn_w *
        (double)c_in * (double)k_h * (double)k_w;
    const size_t bn_elems = n * c_out * bn_h * bn_w;
    if (macs < macs_threshold || bn_elems < out_elems_threshold)
      return false;
  }

  @autoreleasepool {
    id<MTLDevice>      device = _st_be_mps_device();
    id<MTLCommandQueue> queue = _st_be_mps_queue();
    if (!device || !queue) return false;

    const int has_bias    = (bias  != NULL) ? 1 : 0;
    const int has_gamma   = (gamma != NULL) ? 1 : 0;
    const int has_beta    = (beta  != NULL) ? 1 : 0;
    const int do_relu     = apply_relu ? 1 : 0;
    const int pool_type   = (int)pool_params->pool_type;
    const int has_mean_out = (mean != NULL) ? 1 : 0;
    const int has_var_out  = (var  != NULL) ? 1 : 0;

    NSString *cacheKey = [NSString stringWithFormat:
        @"convbnpool:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu"
         ",%zu,%zu,%zu,%zu,%zu,%zu,%d,%d,%d,%d,%d,%d,%d,%.9g",
        n, c_in, h_in, w_in, c_out, k_h, k_w,
        conv_params->stride_h, conv_params->stride_w,
        conv_params->pad_h, conv_params->pad_w,
        conv_params->dilation_h, conv_params->dilation_w,
        pool_params->kernel_h, pool_params->kernel_w,
        pool_params->stride_h, pool_params->stride_w,
        pool_params->pad_h, pool_params->pad_w,
        pool_type, has_bias, has_gamma, has_beta, do_relu,
        has_mean_out, has_var_out, (double)epsilon];

    NSCache *cache = _st_mps_graph_cache();
    NSDictionary *cached = [cache objectForKey:cacheKey];

    MPSGraph *graph;
    MPSGraphTensor *inT, *wT, *biasT, *gammaT, *betaT;
    MPSGraphTensor *resultT, *meanT, *varT;
    MPSGraphExecutable *executable = nil;

    if (cached) {
      graph      = cached[@"graph"];
      executable = cached[@"executable"];
      if ([executable isEqual:[NSNull null]]) executable = nil;
      inT     = cached[@"inT"];
      wT      = cached[@"wT"];
      biasT   = cached[@"biasT"];
      gammaT  = cached[@"gammaT"];
      betaT   = cached[@"betaT"];
      resultT = cached[@"resultT"];
      meanT   = cached[@"meanT"];
      varT    = cached[@"varT"];
      if ([biasT  isEqual:[NSNull null]]) biasT  = nil;
      if ([gammaT isEqual:[NSNull null]]) gammaT = nil;
      if ([betaT  isEqual:[NSNull null]]) betaT  = nil;
    } else {
      graph = [[MPSGraph alloc] init];

      MPSShape *inShape = @[ @(n), @(c_in), @(h_in), @(w_in) ];
      MPSShape *wShape  = @[ @(c_out), @(c_in), @(k_h), @(k_w) ];

      inT = [graph placeholderWithShape:inShape dataType:MPSDataTypeFloat32 name:@"input"];
      wT  = [graph placeholderWithShape:wShape  dataType:MPSDataTypeFloat32 name:@"weight"];

      MPSGraphConvolution2DOpDescriptor *convDesc =
          [MPSGraphConvolution2DOpDescriptor
              descriptorWithStrideInX:(NSUInteger)conv_params->stride_w
                            strideInY:(NSUInteger)conv_params->stride_h
                      dilationRateInX:(NSUInteger)conv_params->dilation_w
                      dilationRateInY:(NSUInteger)conv_params->dilation_h
                               groups:1
                          paddingLeft:(NSUInteger)conv_params->pad_w
                         paddingRight:(NSUInteger)conv_params->pad_w
                           paddingTop:(NSUInteger)conv_params->pad_h
                        paddingBottom:(NSUInteger)conv_params->pad_h
                         paddingStyle:MPSGraphPaddingStyleExplicit
                           dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                        weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

      MPSGraphTensor *convT =
          [graph convolution2DWithSourceTensor:inT weightsTensor:wT
                                    descriptor:convDesc name:@"conv2d"];

      MPSGraphTensor *bnInputT = convT;
      biasT = nil; gammaT = nil; betaT = nil;

      if (bias) {
        biasT = [graph placeholderWithShape:@[ @1, @(c_out), @1, @1 ]
                                   dataType:MPSDataTypeFloat32 name:@"bias"];
        bnInputT = [graph additionWithPrimaryTensor:bnInputT
                                    secondaryTensor:biasT name:@"add_bias"];
      }

      /* BatchNorm */
      NSArray<NSNumber *> *reduceAxes = @[ @0, @2, @3 ];
      meanT = [graph meanOfTensor:bnInputT axes:reduceAxes name:@"mean"];
      MPSGraphTensor *diffT = [graph subtractionWithPrimaryTensor:bnInputT
                                                  secondaryTensor:meanT name:@"diff"];
      varT = [graph meanOfTensor:[graph squareWithTensor:diffT name:@"sq_diff"]
                            axes:reduceAxes name:@"var"];
      MPSGraphTensor *epsT =
          [graph constantWithScalar:(double)epsilon shape:@[ @1 ]
                           dataType:MPSDataTypeFloat32];
      MPSGraphTensor *invStdT =
          [graph reciprocalWithTensor:
              [graph squareRootWithTensor:
                  [graph additionWithPrimaryTensor:varT secondaryTensor:epsT name:@"var_eps"]
                                     name:@"sqrt_var"]
                                    name:@"inv_std"];
      resultT = [graph multiplicationWithPrimaryTensor:diffT
                                       secondaryTensor:invStdT name:@"normed"];

      if (gamma) {
        gammaT = [graph placeholderWithShape:@[ @1, @(c_out), @1, @1 ]
                                    dataType:MPSDataTypeFloat32 name:@"gamma"];
        resultT = [graph multiplicationWithPrimaryTensor:resultT
                                         secondaryTensor:gammaT name:@"scale"];
      }
      if (beta) {
        betaT = [graph placeholderWithShape:@[ @1, @(c_out), @1, @1 ]
                                   dataType:MPSDataTypeFloat32 name:@"beta"];
        resultT = [graph additionWithPrimaryTensor:resultT
                                   secondaryTensor:betaT name:@"shift"];
      }
      if (apply_relu) {
        resultT = [graph reLUWithTensor:resultT name:@"relu"];
      }

      /* Pool op — NCHW layout (axes 2,3 = H,W) */
      MPSGraphPooling2DOpDescriptor *poolDesc =
          [MPSGraphPooling2DOpDescriptor
              descriptorWithKernelWidth:(NSUInteger)pool_params->kernel_w
                           kernelHeight:(NSUInteger)pool_params->kernel_h
                              strideInX:(NSUInteger)pool_params->stride_w
                              strideInY:(NSUInteger)pool_params->stride_h
                           paddingStyle:MPSGraphPaddingStyleExplicit
                             dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
      poolDesc.paddingLeft   = (NSUInteger)pool_params->pad_w;
      poolDesc.paddingRight  = (NSUInteger)pool_params->pad_w;
      poolDesc.paddingTop    = (NSUInteger)pool_params->pad_h;
      poolDesc.paddingBottom = (NSUInteger)pool_params->pad_h;

      if (pool_params->pool_type == ST_POOL_MAX) {
        resultT = [graph maxPooling2DWithSourceTensor:resultT
                                          descriptor:poolDesc name:@"maxpool"];
      } else {
        resultT = [graph avgPooling2DWithSourceTensor:resultT
                                          descriptor:poolDesc name:@"avgpool"];
      }

      /* Compile for inference path. */
      if (!has_mean_out && !has_var_out) {
        MPSShape *bnShape = @[ @1, @(c_out), @1, @1 ];
        NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feedShapes =
            [NSMutableDictionary dictionaryWithCapacity:5];
        feedShapes[inT] = [[MPSGraphShapedType alloc]
            initWithShape:inShape dataType:MPSDataTypeFloat32];
        feedShapes[wT]  = [[MPSGraphShapedType alloc]
            initWithShape:wShape dataType:MPSDataTypeFloat32];
        if (biasT)
          feedShapes[biasT] = [[MPSGraphShapedType alloc]
              initWithShape:bnShape dataType:MPSDataTypeFloat32];
        if (gammaT)
          feedShapes[gammaT] = [[MPSGraphShapedType alloc]
              initWithShape:bnShape dataType:MPSDataTypeFloat32];
        if (betaT)
          feedShapes[betaT] = [[MPSGraphShapedType alloc]
              initWithShape:bnShape dataType:MPSDataTypeFloat32];
        executable = [graph
            compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                        feeds:feedShapes
                targetTensors:@[ resultT ]
             targetOperations:nil
         compilationDescriptor:nil];
      }

      [cache setObject:@{
        @"graph"      : graph,
        @"executable" : executable ?: [NSNull null],
        @"inT"     : inT,
        @"wT"      : wT,
        @"biasT"   : biasT  ?: [NSNull null],
        @"gammaT"  : gammaT ?: [NSNull null],
        @"betaT"   : betaT  ?: [NSNull null],
        @"resultT" : resultT,
        @"meanT"   : meanT,
        @"varT"    : varT,
      } forKey:cacheKey];
    }

    MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSShape *inShape = @[ @(n), @(c_in), @(h_in), @(w_in) ];
    MPSShape *wShape  = @[ @(c_out), @(c_in), @(k_h), @(k_w) ];
    const size_t inBytes = n * c_in * h_in * w_in * sizeof(float);
    const size_t wBytes  = c_out * c_in * k_h * k_w * sizeof(float);

    void *in_mh = input->buf  ? st_buffer_metal_handle(input->buf)  : NULL;
    void *w_mh  = weight->buf ? st_buffer_metal_handle(weight->buf) : NULL;
    MPSGraphTensorData *inData =
        st_mps_make_tensor_data(gDev, input->values,  in_mh, inBytes, inShape);
    MPSGraphTensorData *wData =
        st_mps_make_tensor_data(gDev, weight->values, w_mh,  wBytes,  wShape);

    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
        [NSMutableDictionary dictionaryWithCapacity:5];
    feeds[inT] = inData;
    feeds[wT]  = wData;

    if (bias && biasT) {
      void *b_mh = bias->buf ? st_buffer_metal_handle(bias->buf) : NULL;
      feeds[biasT] = st_mps_make_tensor_data(gDev, bias->values, b_mh,
                         c_out * sizeof(float), @[ @1, @(c_out), @1, @1 ]);
    }
    if (gamma && gammaT) {
      void *g_mh = gamma->buf ? st_buffer_metal_handle(gamma->buf) : NULL;
      feeds[gammaT] = st_mps_make_tensor_data(gDev, gamma->values, g_mh,
                          c_out * sizeof(float), @[ @1, @(c_out), @1, @1 ]);
    }
    if (beta && betaT) {
      void *b_mh = beta->buf ? st_buffer_metal_handle(beta->buf) : NULL;
      feeds[betaT] = st_mps_make_tensor_data(gDev, beta->values, b_mh,
                         c_out * sizeof(float), @[ @1, @(c_out), @1, @1 ]);
    }

    /* ---- Inference path: executable + pre-allocated output MTLBuffer ---- */
    if (executable && output->buf && st_buffer_metal_handle(output->buf)) {
      NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedMap =
          [NSMutableDictionary dictionaryWithDictionary:feeds];
      NSMutableArray<MPSGraphTensorData *> *inputsArray =
          [NSMutableArray arrayWithCapacity:executable.feedTensors.count];
      for (MPSGraphTensor *ft in executable.feedTensors) {
        MPSGraphTensorData *td = feedMap[ft];
        if (!td) return false;
        [inputsArray addObject:td];
      }

      MPSShape *outShape = @[ @(n), @(c_out), @(out_h), @(out_w) ];
      id<MTLBuffer> outBuf =
          (__bridge id<MTLBuffer>)st_buffer_metal_handle(output->buf);
      MPSGraphTensorData *preOutData =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                  shape:outShape
                                               dataType:MPSDataTypeFloat32];

      /* Async encode: commit without waiting; caller uses st_tensor_sync(). */
      id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
      if (!cmdBuf) return false;
      @try {
        [executable encodeToCommandBuffer:cmdBuf
                               inputsArray:inputsArray
                              resultsArray:@[ preOutData ]
                       executionDescriptor:nil];
      } @catch (NSException *exception) {
        return false;
      }
      [cmdBuf commit];
      /* Drain any previously-pending cmd buffer for this output before
         overwriting (prevents orphaned cmd buffers in iterative loops). */
      if (output->buf->_async_cmd_buf) {
        st_buffer_metal_wait(output->buf);
      }
      output->buf->_async_cmd_buf = (__bridge_retained void *)cmdBuf;
      return true;
    }

    /* ---- Training / non-Metal-output path ---- */
    NSMutableArray<MPSGraphTensor *> *targets =
        [NSMutableArray arrayWithObject:resultT];
    if (mean && meanT) [targets addObject:meanT];
    if (var  && varT)  [targets addObject:varT];

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = nil;
    @try {
      results = [graph runWithMTLCommandQueue:queue
                                       feeds:feeds
                               targetTensors:targets
                            targetOperations:nil];
    } @catch (NSException *exception) {
      return false;
    }

    MPSGraphTensorData *outData = results[resultT];
    if (!outData) return false;
    [outData.mpsndarray readBytes:output->values strideBytes:nil];
    if (mean) {
      MPSGraphTensorData *md = results[meanT];
      if (md) [md.mpsndarray readBytes:mean->values strideBytes:nil];
    }
    if (var) {
      MPSGraphTensorData *vd = results[varT];
      if (vd) [vd.mpsndarray readBytes:var->values strideBytes:nil];
    }
    return true;
  }
}

bool st_backend_set_mps_thresholds(size_t pool_threshold,
                                   size_t batchnorm_threshold) {
  if (pool_threshold == 0u || batchnorm_threshold == 0u) {
    return false;
  }

  g_mps_pool_threshold = pool_threshold;
  g_mps_bn_threshold = batchnorm_threshold;
  atomic_store_explicit(&g_mps_thresholds_initialized, true,
                        memory_order_release);
  return true;
}

void st_backend_get_mps_thresholds(size_t *out_pool_threshold,
                                   size_t *out_batchnorm_threshold) {
  st_backend_init_mps_thresholds_once();
  if (out_pool_threshold != NULL) {
    *out_pool_threshold = g_mps_pool_threshold;
  }
  if (out_batchnorm_threshold != NULL) {
    *out_batchnorm_threshold = g_mps_bn_threshold;
  }
}

void st_backend_reload_mps_thresholds_from_env(void) {
  g_mps_pool_threshold = ST_MPS_POOL_THRESHOLD_DEFAULT;
  g_mps_bn_threshold = ST_MPS_BN_THRESHOLD_DEFAULT;

  const char *pool_env = getenv("MMATRIX_ST_POOL_MPS_THRESHOLD");
  if (pool_env != NULL) {
    size_t parsed = 0;
    if (st_backend_parse_positive_size_t(pool_env, &parsed)) {
      g_mps_pool_threshold = parsed;
    }
  }

  const char *bn_env = getenv("MMATRIX_ST_BN_MPS_THRESHOLD");
  if (bn_env != NULL) {
    size_t parsed = 0;
    if (st_backend_parse_positive_size_t(bn_env, &parsed)) {
      g_mps_bn_threshold = parsed;
    }
  }

  atomic_store_explicit(&g_mps_thresholds_initialized, true,
                        memory_order_release);
}

bool st_backend_set_conv_mps_async(bool enabled) {
  atomic_store_explicit(&g_mps_conv_async_enabled, enabled,
                        memory_order_release);
  atomic_store_explicit(&g_mps_conv_async_initialized, true,
                        memory_order_release);
  return true;
}

bool st_backend_get_conv_mps_async(void) {
  st_backend_init_conv_async_once();
  return atomic_load_explicit(&g_mps_conv_async_enabled,
                              memory_order_acquire);
}

void st_backend_reload_conv_mps_async_from_env(void) {
  bool async_enabled = ST_MPS_CONV_ASYNC_DEFAULT;
  const char *async_env = getenv("MMATRIX_ST_CONV_MPS_ASYNC");
  if (async_env != NULL) {
    bool parsed = false;
    if (st_backend_parse_bool(async_env, &parsed)) {
      async_enabled = parsed;
    }
  }

  atomic_store_explicit(&g_mps_conv_async_enabled, async_enabled,
                        memory_order_release);
  atomic_store_explicit(&g_mps_conv_async_initialized, true,
                        memory_order_release);
}

bool st_backend_set_conv_mps_nhwc(bool enabled) {
  atomic_store_explicit(&g_mps_conv_nhwc_enabled, enabled,
                        memory_order_release);
  atomic_store_explicit(&g_mps_conv_nhwc_initialized, true,
                        memory_order_release);
  return true;
}

bool st_backend_get_conv_mps_nhwc(void) {
  st_backend_init_conv_nhwc_once();
  return atomic_load_explicit(&g_mps_conv_nhwc_enabled, memory_order_acquire);
}

void st_backend_reload_conv_mps_nhwc_from_env(void) {
  bool nhwc_enabled = ST_MPS_CONV_NHWC_DEFAULT;
  const char *env = getenv("MMATRIX_ST_CONV_MPS_NHWC");
  if (env != NULL) {
    bool parsed = false;
    if (st_backend_parse_bool(env, &parsed)) {
      nhwc_enabled = parsed;
    }
  }

  atomic_store_explicit(&g_mps_conv_nhwc_enabled, nhwc_enabled,
                        memory_order_release);
  atomic_store_explicit(&g_mps_conv_nhwc_initialized, true,
                        memory_order_release);
}

/* ================================================================== */
/*  Singleton vtable                                                   */
/* ================================================================== */

static const StBackend s_mps_backend = {
    .name               = "mps",
    .supports_op        = mps_supports_op,
    .conv2d_forward     = mps_conv2d_forward,
    .maxpool2d_forward  = mps_maxpool2d_forward,
    .avgpool2d_forward  = mps_avgpool2d_forward,
    .batchnorm2d_forward = mps_batchnorm2d_forward,
};

const StBackend *st_backend_mps(void) { return &s_mps_backend; }

/* ================================================================== */
/*  Conv2D warmup: pre-populate MPSGraph cache for a given shape       */
/* ================================================================== */

void st_backend_mps_warmup_conv2d(size_t n, size_t c_in, size_t h, size_t w,
                                   size_t c_out, size_t kh, size_t kw,
                                   size_t sh, size_t sw, size_t ph, size_t pw,
                                   size_t dh, size_t dw) {
  size_t out_h = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
  size_t out_w = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

  size_t shape_in[4]  = {n, c_in,  h,    w};
  size_t shape_w[4]   = {c_out, c_in, kh, kw};
  size_t shape_out[4] = {n, c_out, out_h, out_w};

  FloatTensor *tin  = st_create(4, shape_in);
  FloatTensor *tw   = st_create(4, shape_w);
  FloatTensor *tout = st_create(4, shape_out);
  if (!tin || !tw || !tout) {
    st_destroy(tin); st_destroy(tw); st_destroy(tout);
    return;
  }

  StConv2dParams params = {
    .stride_h   = sh, .stride_w   = sw,
    .pad_h      = ph, .pad_w      = pw,
    .dilation_h = dh, .dilation_w = dw,
    .backend    = ST_CONV_BACKEND_MPS,
  };

  mps_conv2d_forward(tin, tw, NULL, &params, tout);

  st_destroy(tin); st_destroy(tw); st_destroy(tout);
}
