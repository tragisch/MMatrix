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
#import "st_conv.h"
#import "st_mps.h"
#import "sm_mps.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

/* ---- Shared device / queue (reuse from sm_mps) ---- */

static id<MTLDevice> _st_be_mps_device(void) {
  return (__bridge id<MTLDevice>)mps_get_shared_device();
}

static id<MTLCommandQueue> _st_be_mps_queue(void) {
  return (__bridge id<MTLCommandQueue>)mps_get_shared_command_queue();
}

/* ---- Helper: create MPSGraphTensorData with zero-copy when possible ---- */

static MPSGraphTensorData *_st_be_make_tensor_data(MPSGraphDevice *gDev,
                                                   const float *data,
                                                   void *metal_handle,
                                                   size_t bytes,
                                                   MPSShape *shape) {
  if (metal_handle) {
    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)metal_handle;
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:mtl_buf
                                                  shape:shape
                                               dataType:MPSDataTypeFloat32];
  }
  return [[MPSGraphTensorData alloc]
      initWithDevice:gDev
                data:[NSData dataWithBytes:data length:bytes]
               shape:shape
            dataType:MPSDataTypeFloat32];
}

/* ================================================================== */
/*  Thresholds                                                         */
/* ================================================================== */

#define ST_MPS_POOL_THRESHOLD   4096
#define ST_MPS_BN_THRESHOLD     4096

/* Conv thresholds are read from st_conv via the public threshold API. */

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

/* ================================================================== */
/*  supports_op                                                        */
/* ================================================================== */

static bool mps_supports_op(StOp op, size_t numel,
                            StBufferType buf_type __attribute__((unused))) {
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
      return numel >= ST_MPS_POOL_THRESHOLD;
    case ST_OP_BATCHNORM2D_FORWARD:
      return numel >= ST_MPS_BN_THRESHOLD;
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

  /* ---- Cache lookup by shape signature ---- */
  const int has_bias = (bias != NULL) ? 1 : 0;
  NSString *cacheKey = [NSString stringWithFormat:
      @"conv:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d",
      n, c_in, h_in, w_in, c_out, k_h, k_w,
      params->stride_h, params->stride_w,
      params->pad_h, params->pad_w,
      params->dilation_h, params->dilation_w, has_bias];

  NSCache *cache = _st_mps_graph_cache();
  NSDictionary *cached = [cache objectForKey:cacheKey];

  MPSGraph *graph;
  MPSGraphTensor *inT, *wT, *biasT, *resultT;

  if (cached) {
    /* Cache hit — reuse compiled graph. */
    graph   = cached[@"graph"];
    inT     = cached[@"inT"];
    wT      = cached[@"wT"];
    biasT   = cached[@"biasT"];   /* may be NSNull */
    resultT = cached[@"resultT"];
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

    resultT = convT;
    biasT   = nil;

    if (bias) {
      biasT   = [graph placeholderWithShape:@[ @1, @(c_out), @1, @1 ]
                                   dataType:MPSDataTypeFloat32
                                       name:@"bias"];
      resultT = [graph additionWithPrimaryTensor:convT
                                 secondaryTensor:biasT
                                            name:@"add_bias"];
    }

    /* Store into cache. */
    [cache setObject:@{
      @"graph"   : graph,
      @"inT"     : inT,
      @"wT"      : wT,
      @"biasT"   : biasT   ?: [NSNull null],
      @"resultT" : resultT,
    } forKey:cacheKey];
  }

  /* ---- Prepare feed data ---- */
  MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];

  void *in_mh  = input->buf  ? st_buffer_metal_handle(input->buf)  : NULL;
  void *w_mh   = weight->buf ? st_buffer_metal_handle(weight->buf) : NULL;

  MPSShape *inShape = @[ @(n), @(c_in), @(h_in), @(w_in) ];
  MPSShape *wShape  = @[ @(c_out), @(c_in), @(k_h), @(k_w) ];

  const size_t inBytes = n * c_in * h_in * w_in * sizeof(float);
  const size_t wBytes  = c_out * c_in * k_h * k_w * sizeof(float);

  MPSGraphTensorData *inData = _st_be_make_tensor_data(gDev, input->values,
                                                        in_mh, inBytes,
                                                        inShape);
  MPSGraphTensorData *wData  = _st_be_make_tensor_data(gDev, weight->values,
                                                        w_mh, wBytes, wShape);

  NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
      [NSMutableDictionary dictionaryWithCapacity:3];
  feeds[inT] = inData;
  feeds[wT]  = wData;

  if (bias && biasT) {
    void *b_mh = bias->buf ? st_buffer_metal_handle(bias->buf) : NULL;
    const size_t bBytes = c_out * sizeof(float);
    MPSGraphTensorData *bData = _st_be_make_tensor_data(
        gDev, bias->values, b_mh, bBytes, @[ @1, @(c_out), @1, @1 ]);
    feeds[biasT] = bData;
  }

  /* ---- Run graph synchronously ---- */
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
      epsilon, output->values, mean->values, var->values);
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
