/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#import "st_mps.h"
#import "sm_mps.h"

#import <Metal/Metal.h>
#import <string.h>

/* ---- Shared device / queue (reuse from sm_mps) ---- */

static id<MTLDevice> _st_mps_device(void) {
  return (__bridge id<MTLDevice>)mps_get_shared_device();
}

static id<MTLCommandQueue> _st_mps_queue(void) {
  return (__bridge id<MTLCommandQueue>)mps_get_shared_command_queue();
}

/* ---- MPSGraph cache (NSCache — thread-safe, auto-evict LRU) ---- */

static NSCache<NSString *, NSDictionary *> *_st_mps_pool_cache(void) {
  static NSCache<NSString *, NSDictionary *> *cache = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    cache = [[NSCache alloc] init];
    cache.countLimit = 16;
  });
  return cache;
}

/* ================================================================== */
/*  MPS MaxPool2D (NCHW layout)                                       */
/* ================================================================== */

bool st_maxpool2d_mps(const float *input, void *input_metal_handle,
                      size_t n, size_t c, size_t h,
                      size_t w, size_t kernel_h, size_t kernel_w,
                      size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, float *output, size_t out_h,
                      size_t out_w) {
  if (!input || !output) return false;
  if (n == 0 || c == 0 || h == 0 || w == 0) return false;

  @autoreleasepool {

  id<MTLDevice> device = _st_mps_device();
  id<MTLCommandQueue> queue = _st_mps_queue();
  if (!device || !queue) return false;

  /* ---- Cache lookup ---- */
  NSString *cacheKey = [NSString stringWithFormat:
      @"maxpool:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu",
      n, c, h, w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w];

  NSCache *cache = _st_mps_pool_cache();
  NSDictionary *cached = [cache objectForKey:cacheKey];

  MPSGraph *graph;
  MPSGraphTensor *inT, *resultT;

  if (cached) {
    graph   = cached[@"graph"];
    inT     = cached[@"inT"];
    resultT = cached[@"resultT"];
  } else {
    graph = [[MPSGraph alloc] init];

    MPSShape *shape = @[ @(n), @(c), @(h), @(w) ];
    inT = [graph placeholderWithShape:shape
                             dataType:MPSDataTypeFloat32
                                 name:@"input"];

    MPSGraphPooling2DOpDescriptor *poolDesc =
        [MPSGraphPooling2DOpDescriptor
            descriptorWithKernelWidth:(NSUInteger)kernel_w
                        kernelHeight:(NSUInteger)kernel_h
                           strideInX:(NSUInteger)stride_w
                           strideInY:(NSUInteger)stride_h
                        paddingStyle:MPSGraphPaddingStyleExplicit
                          dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
    poolDesc.paddingLeft   = (NSUInteger)pad_w;
    poolDesc.paddingRight  = (NSUInteger)pad_w;
    poolDesc.paddingTop    = (NSUInteger)pad_h;
    poolDesc.paddingBottom = (NSUInteger)pad_h;

    resultT = [graph maxPooling2DWithSourceTensor:inT
                                       descriptor:poolDesc
                                             name:@"maxpool"];

    [cache setObject:@{
      @"graph"   : graph,
      @"inT"     : inT,
      @"resultT" : resultT,
    } forKey:cacheKey];
  }

  /* Feed data */
  MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];
  MPSShape *inShape = @[ @(n), @(c), @(h), @(w) ];
  const size_t inBytes = n * c * h * w * sizeof(float);
  MPSGraphTensorData *inData = st_mps_make_tensor_data(gDev, input,
                                                       input_metal_handle,
                                                       inBytes, inShape);

  NSDictionary *feeds = @{inT : inData};

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

  [outData.mpsndarray readBytes:output strideBytes:nil];
  return true;

  } // @autoreleasepool
}

/* ================================================================== */
/*  MPS AvgPool2D (NCHW layout)                                       */
/* ================================================================== */

bool st_avgpool2d_mps(const float *input, void *input_metal_handle,
                      size_t n, size_t c, size_t h,
                      size_t w, size_t kernel_h, size_t kernel_w,
                      size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, float *output, size_t out_h,
                      size_t out_w) {
  if (!input || !output) return false;
  if (n == 0 || c == 0 || h == 0 || w == 0) return false;

  @autoreleasepool {

  id<MTLDevice> device = _st_mps_device();
  id<MTLCommandQueue> queue = _st_mps_queue();
  if (!device || !queue) return false;

  /* ---- Cache lookup ---- */
  NSString *cacheKey = [NSString stringWithFormat:
      @"avgpool:%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu",
      n, c, h, w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w];

  NSCache *cache = _st_mps_pool_cache();
  NSDictionary *cached = [cache objectForKey:cacheKey];

  MPSGraph *graph;
  MPSGraphTensor *inT, *resultT;

  if (cached) {
    graph   = cached[@"graph"];
    inT     = cached[@"inT"];
    resultT = cached[@"resultT"];
  } else {
    graph = [[MPSGraph alloc] init];

    MPSShape *shape = @[ @(n), @(c), @(h), @(w) ];
    inT = [graph placeholderWithShape:shape
                             dataType:MPSDataTypeFloat32
                                 name:@"input"];

    MPSGraphPooling2DOpDescriptor *poolDesc =
        [MPSGraphPooling2DOpDescriptor
            descriptorWithKernelWidth:(NSUInteger)kernel_w
                        kernelHeight:(NSUInteger)kernel_h
                           strideInX:(NSUInteger)stride_w
                           strideInY:(NSUInteger)stride_h
                        paddingStyle:MPSGraphPaddingStyleExplicit
                          dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
    poolDesc.paddingLeft   = (NSUInteger)pad_w;
    poolDesc.paddingRight  = (NSUInteger)pad_w;
    poolDesc.paddingTop    = (NSUInteger)pad_h;
    poolDesc.paddingBottom = (NSUInteger)pad_h;

    resultT = [graph avgPooling2DWithSourceTensor:inT
                                       descriptor:poolDesc
                                             name:@"avgpool"];

    [cache setObject:@{
      @"graph"   : graph,
      @"inT"     : inT,
      @"resultT" : resultT,
    } forKey:cacheKey];
  }

  /* Feed data */
  MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];
  MPSShape *inShape = @[ @(n), @(c), @(h), @(w) ];
  const size_t inBytes = n * c * h * w * sizeof(float);
  MPSGraphTensorData *inData = st_mps_make_tensor_data(gDev, input,
                                                       input_metal_handle,
                                                       inBytes, inShape);

  NSDictionary *feeds = @{inT : inData};

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

  [outData.mpsndarray readBytes:output strideBytes:nil];
  return true;

  } // @autoreleasepool
}

/* ================================================================== */
/*  MPS BatchNorm2D Forward (NCHW layout)                             */
/* ================================================================== */

bool st_batchnorm2d_forward_mps(const float *input, void *input_metal_handle,
                                size_t n, size_t c,
                                size_t h, size_t w, const float *gamma,
                                const float *beta, float epsilon,
                                float *output, float *mean_out,
                                float *var_out) {
  if (!input || !output || !mean_out || !var_out) return false;
  if (n == 0 || c == 0 || h == 0 || w == 0) return false;

  @autoreleasepool {

  id<MTLDevice> device = _st_mps_device();
  id<MTLCommandQueue> queue = _st_mps_queue();
  if (!device || !queue) return false;

  /* ---- Cache lookup ---- */
  const int has_gamma = (gamma != NULL) ? 1 : 0;
  const int has_beta  = (beta  != NULL) ? 1 : 0;
  NSString *cacheKey = [NSString stringWithFormat:
      @"bn:%zu,%zu,%zu,%zu,%d,%d",
      n, c, h, w, has_gamma, has_beta];

  NSCache *cache = _st_mps_pool_cache();
  NSDictionary *cached = [cache objectForKey:cacheKey];

  MPSGraph *graph;
  MPSGraphTensor *inT, *gammaT, *betaT, *resultT, *meanT, *varT;

  if (cached) {
    graph   = cached[@"graph"];
    inT     = cached[@"inT"];
    gammaT  = cached[@"gammaT"];
    betaT   = cached[@"betaT"];
    resultT = cached[@"resultT"];
    meanT   = cached[@"meanT"];
    varT    = cached[@"varT"];
    if ([gammaT isEqual:[NSNull null]]) gammaT = nil;
    if ([betaT  isEqual:[NSNull null]]) betaT  = nil;
  } else {
    graph = [[MPSGraph alloc] init];

    MPSShape *inShape = @[ @(n), @(c), @(h), @(w) ];
    inT = [graph placeholderWithShape:inShape
                             dataType:MPSDataTypeFloat32
                                 name:@"input"];

    /* Compute mean and variance along N, H, W (axes 0, 2, 3). */
    NSArray<NSNumber *> *reduceAxes = @[ @0, @2, @3 ];

    meanT = [graph meanOfTensor:inT axes:reduceAxes name:@"mean"];

    MPSGraphTensor *diffT =
        [graph subtractionWithPrimaryTensor:inT
                            secondaryTensor:meanT
                                       name:@"diff"];

    MPSGraphTensor *sqDiffT =
        [graph squareWithTensor:diffT name:@"sq_diff"];

    varT = [graph meanOfTensor:sqDiffT axes:reduceAxes name:@"var"];

    /* Normalize: (x - mean) / sqrt(var + epsilon) */
    MPSGraphTensor *epsT =
        [graph constantWithScalar:(double)epsilon
                            shape:@[ @1 ]
                         dataType:MPSDataTypeFloat32];

    MPSGraphTensor *varPlusEpsT =
        [graph additionWithPrimaryTensor:varT
                         secondaryTensor:epsT
                                    name:@"var_eps"];

    MPSGraphTensor *sqrtVarT =
        [graph squareRootWithTensor:varPlusEpsT
                               name:@"sqrt_var"];

    MPSGraphTensor *invStdT =
        [graph reciprocalWithTensor:sqrtVarT
                               name:@"inv_std"];

    MPSGraphTensor *normedT =
        [graph multiplicationWithPrimaryTensor:diffT
                               secondaryTensor:invStdT
                                          name:@"normed"];

    /* Scale and shift: gamma * normed + beta */
    resultT = normedT;
    gammaT  = nil;
    betaT   = nil;

    if (gamma) {
      gammaT = [graph placeholderWithShape:@[ @1, @(c), @1, @1 ]
                                  dataType:MPSDataTypeFloat32
                                      name:@"gamma"];
      resultT = [graph multiplicationWithPrimaryTensor:resultT
                                       secondaryTensor:gammaT
                                                  name:@"scale"];
    }

    if (beta) {
      betaT = [graph placeholderWithShape:@[ @1, @(c), @1, @1 ]
                                 dataType:MPSDataTypeFloat32
                                     name:@"beta"];
      resultT = [graph additionWithPrimaryTensor:resultT
                                 secondaryTensor:betaT
                                            name:@"shift"];
    }

    [cache setObject:@{
      @"graph"   : graph,
      @"inT"     : inT,
      @"gammaT"  : gammaT  ?: [NSNull null],
      @"betaT"   : betaT   ?: [NSNull null],
      @"resultT" : resultT,
      @"meanT"   : meanT,
      @"varT"    : varT,
    } forKey:cacheKey];
  }

  /* Feed data */
  MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];
  MPSShape *inShape = @[ @(n), @(c), @(h), @(w) ];
  const size_t inBytes = n * c * h * w * sizeof(float);

  MPSGraphTensorData *inData = st_mps_make_tensor_data(gDev, input,
                                                       input_metal_handle,
                                                       inBytes, inShape);

  NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
      [NSMutableDictionary dictionaryWithCapacity:3];
  feeds[inT] = inData;

  if (gamma && gammaT) {
    const size_t gBytes = c * sizeof(float);
    MPSGraphTensorData *gData = st_mps_make_tensor_data(
        gDev, gamma, NULL, gBytes, @[ @1, @(c), @1, @1 ]);
    feeds[gammaT] = gData;
  }

  if (beta && betaT) {
    const size_t bBytes = c * sizeof(float);
    MPSGraphTensorData *bData = st_mps_make_tensor_data(
        gDev, beta, NULL, bBytes, @[ @1, @(c), @1, @1 ]);
    feeds[betaT] = bData;
  }

  /* Run graph: get output, mean, and variance. */
  NSMutableArray<MPSGraphTensor *> *targets =
      [NSMutableArray arrayWithObjects:resultT, meanT, varT, nil];

  NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = nil;
  @try {
    results = [graph runWithMTLCommandQueue:queue
                                     feeds:feeds
                             targetTensors:targets
                          targetOperations:nil];
  } @catch (NSException *exception) {
    return false;
  }

  /* Read output */
  MPSGraphTensorData *outData = results[resultT];
  if (!outData) return false;
  [outData.mpsndarray readBytes:output strideBytes:nil];

  /* Read mean [1, C, 1, 1] → flatten to [C] */
  MPSGraphTensorData *meanData = results[meanT];
  if (meanData) {
    [meanData.mpsndarray readBytes:mean_out strideBytes:nil];
  }

  /* Read variance [1, C, 1, 1] → flatten to [C] */
  MPSGraphTensorData *varData = results[varT];
  if (varData) {
    [varData.mpsndarray readBytes:var_out strideBytes:nil];
  }

  return true;

  } // @autoreleasepool
}
