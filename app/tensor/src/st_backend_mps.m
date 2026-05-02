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

static bool st_mps_tensor_is_valid(const FloatTensor *t, size_t ndim) {
  return t != NULL && t->values != NULL && t->ndim == ndim;
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

bool st_backend_conv2d_batchnorm2d_forward_mps(
    const FloatTensor *input, const FloatTensor *weight,
    const FloatTensor *bias, const StConv2dParams *params,
    const FloatTensor *gamma, const FloatTensor *beta, float epsilon,
    FloatTensor *output, FloatTensor *mean, FloatTensor *var) {
  if (!input || !weight || !params || !output || !mean || !var) return false;
  if (params->backend != ST_CONV_BACKEND_AUTO &&
      params->backend != ST_CONV_BACKEND_MPS) {
    return false;
  }

  if (input->dtype != ST_DTYPE_F32 || weight->dtype != ST_DTYPE_F32 ||
      output->dtype != ST_DTYPE_F32 || mean->dtype != ST_DTYPE_F32 ||
      var->dtype != ST_DTYPE_F32) {
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
      !st_mps_tensor_is_valid(mean, 1) ||
      !st_mps_tensor_is_valid(var, 1)) {
    return false;
  }
  if (!st_is_contiguous(input) || !st_is_contiguous(weight) ||
      !st_is_contiguous(output) || !st_is_contiguous(mean) ||
      !st_is_contiguous(var)) {
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
  if (mean->shape[0] != c_out || var->shape[0] != c_out) return false;

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
    NSString *cacheKey = [NSString stringWithFormat:
        @"convbn:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%d,%d,%d,%.9g",
        n, c_in, h_in, w_in, c_out, k_h, k_w,
        params->stride_h, params->stride_w,
        params->pad_h, params->pad_w,
        params->dilation_h, params->dilation_w,
        has_bias, has_gamma, has_beta, (double)epsilon];

    NSCache *cache = _st_mps_graph_cache();
    NSDictionary *cached = [cache objectForKey:cacheKey];

    MPSGraph *graph;
    MPSGraphTensor *inT, *wT, *biasT, *gammaT, *betaT;
    MPSGraphTensor *resultT, *meanT, *varT;

    if (cached) {
      graph = cached[@"graph"];
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

      [cache setObject:@{
        @"graph" : graph,
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

    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = nil;
    @try {
      results = [graph runWithMTLCommandQueue:queue
                                       feeds:feeds
                               targetTensors:@[ resultT, meanT, varT ]
                            targetOperations:nil];
    } @catch (NSException *exception) {
      return false;
    }

    MPSGraphTensorData *outData = results[resultT];
    MPSGraphTensorData *meanData = results[meanT];
    MPSGraphTensorData *varData = results[varT];
    if (!outData || !meanData || !varData) return false;

    [outData.mpsndarray readBytes:output->values strideBytes:nil];
    [meanData.mpsndarray readBytes:mean->values strideBytes:nil];
    [varData.mpsndarray readBytes:var->values strideBytes:nil];
    return true;
  }
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
