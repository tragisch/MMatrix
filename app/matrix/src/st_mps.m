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

/* ================================================================== */
/*  MPS MaxPool2D (NCHW layout)                                       */
/* ================================================================== */

bool st_maxpool2d_mps(const float *input, size_t n, size_t c, size_t h,
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

  MPSGraph *graph = [[MPSGraph alloc] init];

  MPSShape *inShape = @[ @(n), @(c), @(h), @(w) ];
  MPSGraphTensor *inT =
      [graph placeholderWithShape:inShape
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

  MPSGraphTensor *resultT =
      [graph maxPooling2DWithSourceTensor:inT
                               descriptor:poolDesc
                                     name:@"maxpool"];

  /* Feed data */
  MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];
  const size_t inBytes = n * c * h * w * sizeof(float);
  MPSGraphTensorData *inData = [[MPSGraphTensorData alloc]
      initWithDevice:gDev
                data:[NSData dataWithBytes:input length:inBytes]
               shape:inShape
            dataType:MPSDataTypeFloat32];

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

bool st_avgpool2d_mps(const float *input, size_t n, size_t c, size_t h,
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

  MPSGraph *graph = [[MPSGraph alloc] init];

  MPSShape *inShape = @[ @(n), @(c), @(h), @(w) ];
  MPSGraphTensor *inT =
      [graph placeholderWithShape:inShape
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

  MPSGraphTensor *resultT =
      [graph avgPooling2DWithSourceTensor:inT
                               descriptor:poolDesc
                                     name:@"avgpool"];

  /* Feed data */
  MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];
  const size_t inBytes = n * c * h * w * sizeof(float);
  MPSGraphTensorData *inData = [[MPSGraphTensorData alloc]
      initWithDevice:gDev
                data:[NSData dataWithBytes:input length:inBytes]
               shape:inShape
            dataType:MPSDataTypeFloat32];

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

bool st_batchnorm2d_forward_mps(const float *input, size_t n, size_t c,
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

  MPSGraph *graph = [[MPSGraph alloc] init];

  MPSShape *inShape = @[ @(n), @(c), @(h), @(w) ];
  MPSGraphTensor *inT =
      [graph placeholderWithShape:inShape
                         dataType:MPSDataTypeFloat32
                             name:@"input"];

  /* Compute mean and variance along N, H, W (axes 0, 2, 3). */
  NSArray<NSNumber *> *reduceAxes = @[ @0, @2, @3 ];

  MPSGraphTensor *meanT =
      [graph meanOfTensor:inT axes:reduceAxes name:@"mean"];

  MPSGraphTensor *diffT =
      [graph subtractionWithPrimaryTensor:inT
                          secondaryTensor:meanT
                                     name:@"diff"];

  MPSGraphTensor *sqDiffT =
      [graph squareWithTensor:diffT name:@"sq_diff"];

  MPSGraphTensor *varT =
      [graph meanOfTensor:sqDiffT axes:reduceAxes name:@"var"];

  /* Normalize: (x - mean) / sqrt(var + epsilon) */
  MPSGraphTensor *epsT =
      [graph constantWithScalar:(double)epsilon
                          shape:@[ @1 ]
                       dataType:MPSDataTypeFloat32];

  MPSGraphTensor *varPlusEpsT =
      [graph additionWithPrimaryTensor:varT
                       secondaryTensor:epsT
                                  name:@"var_eps"];

  MPSGraphTensor *invStdT =
      [graph reciprocalOfSquareRootWithTensor:varPlusEpsT
                                         name:@"inv_std"];

  MPSGraphTensor *normedT =
      [graph multiplicationWithPrimaryTensor:diffT
                             secondaryTensor:invStdT
                                        name:@"normed"];

  /* Scale and shift: gamma * normed + beta */
  MPSGraphTensor *resultT = normedT;

  MPSGraphTensor *gammaT = nil;
  MPSGraphTensor *betaT = nil;

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

  /* Feed data */
  MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];
  const size_t inBytes = n * c * h * w * sizeof(float);

  MPSGraphTensorData *inData = [[MPSGraphTensorData alloc]
      initWithDevice:gDev
                data:[NSData dataWithBytes:input length:inBytes]
               shape:inShape
            dataType:MPSDataTypeFloat32];

  NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
      [NSMutableDictionary dictionaryWithCapacity:3];
  feeds[inT] = inData;

  if (gamma && gammaT) {
    const size_t gBytes = c * sizeof(float);
    MPSGraphTensorData *gData = [[MPSGraphTensorData alloc]
        initWithDevice:gDev
                  data:[NSData dataWithBytes:gamma length:gBytes]
                 shape:@[ @1, @(c), @1, @1 ]
              dataType:MPSDataTypeFloat32];
    feeds[gammaT] = gData;
  }

  if (beta && betaT) {
    const size_t bBytes = c * sizeof(float);
    MPSGraphTensorData *bData = [[MPSGraphTensorData alloc]
        initWithDevice:gDev
                  data:[NSData dataWithBytes:beta length:bBytes]
                 shape:@[ @1, @(c), @1, @1 ]
              dataType:MPSDataTypeFloat32];
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
    float *mean_buf = (float *)malloc(c * sizeof(float));
    if (mean_buf) {
      [meanData.mpsndarray readBytes:mean_buf strideBytes:nil];
      memcpy(mean_out, mean_buf, c * sizeof(float));
      free(mean_buf);
    }
  }

  /* Read variance [1, C, 1, 1] → flatten to [C] */
  MPSGraphTensorData *varData = results[varT];
  if (varData) {
    float *var_buf = (float *)malloc(c * sizeof(float));
    if (var_buf) {
      [varData.mpsndarray readBytes:var_buf strideBytes:nil];
      memcpy(var_out, var_buf, c * sizeof(float));
      free(var_buf);
    }
  }

  return true;

  } // @autoreleasepool
}
