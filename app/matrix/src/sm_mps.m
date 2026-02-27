/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#import "sm_mps.h"

static id<MTLDevice> _mps_shared_device(void) {
  static id<MTLDevice> device = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    device = MTLCreateSystemDefaultDevice();
  });
  return device;
}

static id<MTLCommandQueue> _mps_shared_command_queue(void) {
  static id<MTLCommandQueue> queue = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    queue = [_mps_shared_device() newCommandQueue];
  });
  return queue;
}

void *mps_get_shared_device(void) {
  return (__bridge void *)_mps_shared_device();
}

void *mps_get_shared_command_queue(void) {
  return (__bridge void *)_mps_shared_command_queue();
}

bool mps_matrix_multiply_ex(const float *mat1, size_t rows1, size_t cols1,
                            bool transpose_left, const float *mat2,
                            size_t rows2, size_t cols2, bool transpose_right,
                            float alpha, float beta, float *result,
                            size_t result_rows, size_t result_cols) {
  if (!mat1 || !mat2 || !result) {
    return false;
  }

  size_t left_rows = transpose_left ? cols1 : rows1;
  size_t left_cols = transpose_left ? rows1 : cols1;
  size_t right_rows = transpose_right ? cols2 : rows2;
  size_t right_cols = transpose_right ? rows2 : cols2;

  if (left_cols != right_rows) {
    return false;
  }
  if (result_rows != left_rows || result_cols != right_cols) {
    return false;
  }

  @autoreleasepool {

  id<MTLDevice> device = _mps_shared_device();
  if (!device) {
    return false;
  }

  id<MTLCommandQueue> commandQueue = _mps_shared_command_queue();
  if (!commandQueue) {
    return false;
  }

  MPSMatrixDescriptor *descA =
      [MPSMatrixDescriptor matrixDescriptorWithRows:rows1
                                            columns:cols1
                                           rowBytes:cols1 * sizeof(float)
                                           dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *descB =
      [MPSMatrixDescriptor matrixDescriptorWithRows:rows2
                                            columns:cols2
                                           rowBytes:cols2 * sizeof(float)
                                           dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *descC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:result_rows
                                            columns:result_cols
                                           rowBytes:result_cols * sizeof(float)
                                           dataType:MPSDataTypeFloat32];

  id<MTLBuffer> bufferA = [device newBufferWithBytes:mat1
                                               length:rows1 * cols1 * sizeof(float)
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferB = [device newBufferWithBytes:mat2
                                               length:rows2 * cols2 * sizeof(float)
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufferC = nil;
  size_t c_bytes = result_rows * result_cols * sizeof(float);
  if (beta != 0.0f) {
    bufferC = [device newBufferWithBytes:result
                                  length:c_bytes
                                 options:MTLResourceStorageModeShared];
  } else {
    bufferC = [device newBufferWithLength:c_bytes
                                  options:MTLResourceStorageModeShared];
  }

  if (!bufferA || !bufferB || !bufferC) {
    return false;
  }

  MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
  MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
  MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

  MPSMatrixMultiplication *mm =
      [[MPSMatrixMultiplication alloc] initWithDevice:device
                                        transposeLeft:(BOOL)transpose_left
                                       transposeRight:(BOOL)transpose_right
                                           resultRows:result_rows
                                        resultColumns:result_cols
                                      interiorColumns:left_cols
                                                alpha:alpha
                                                 beta:beta];

  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  if (!commandBuffer) {
    return false;
  }

  [mm encodeToCommandBuffer:commandBuffer
                 leftMatrix:matrixA
                rightMatrix:matrixB
               resultMatrix:matrixC];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  if (commandBuffer.status == MTLCommandBufferStatusError) {
    return false;
  }

  memcpy(result, matrixC.data.contents, c_bytes);
  return true;

  } // @autoreleasepool
}

bool mps_matrix_multiply(const float *mat1, size_t rows1, size_t cols1,
                         const float *mat2, size_t rows2, size_t cols2,
                         float *result) {
  return mps_matrix_multiply_ex(mat1, rows1, cols1, false, mat2, rows2, cols2,
                                false, 1.0f, 0.0f, result, rows1, cols2);
}

/* ------------------------------------------------------------------ */
/*  MPS Conv2D via MPSGraph (macOS 11+)                               */
/* ------------------------------------------------------------------ */

bool mps_conv2d_nchw(const float *input, size_t n,
                     size_t c_in, size_t h_in, size_t w_in,
                     const float *weight, size_t c_out,
                     size_t k_h, size_t k_w, const float *bias,
                     size_t stride_h, size_t stride_w,
                     size_t pad_h, size_t pad_w,
                     size_t dil_h, size_t dil_w,
                     float *output, size_t h_out, size_t w_out) {
  if (!input || !weight || !output) {
    return false;
  }
  if (n == 0 || c_in == 0 || c_out == 0 || h_in == 0 || w_in == 0) {
    return false;
  }
  if (k_h == 0 || k_w == 0 || h_out == 0 || w_out == 0) {
    return false;
  }

  @autoreleasepool {

  id<MTLDevice> device = _mps_shared_device();
  if (!device) {
    return false;
  }

  id<MTLCommandQueue> queue = _mps_shared_command_queue();
  if (!queue) {
    return false;
  }

  /* ---- Build the MPSGraph ---- */
  MPSGraph *graph = [[MPSGraph alloc] init];

  MPSShape *inShape = @[ @(n), @(c_in), @(h_in), @(w_in) ];
  MPSShape *wShape  = @[ @(c_out), @(c_in), @(k_h), @(k_w) ];

  MPSGraphTensor *inT = [graph placeholderWithShape:inShape
                                           dataType:MPSDataTypeFloat32
                                               name:@"input"];
  MPSGraphTensor *wT  = [graph placeholderWithShape:wShape
                                           dataType:MPSDataTypeFloat32
                                               name:@"weight"];

  MPSGraphConvolution2DOpDescriptor *convDesc =
      [MPSGraphConvolution2DOpDescriptor
          descriptorWithStrideInX:(NSUInteger)stride_w
                        strideInY:(NSUInteger)stride_h
                  dilationRateInX:(NSUInteger)dil_w
                  dilationRateInY:(NSUInteger)dil_h
                           groups:1
                      paddingLeft:(NSUInteger)pad_w
                     paddingRight:(NSUInteger)pad_w
                       paddingTop:(NSUInteger)pad_h
                    paddingBottom:(NSUInteger)pad_h
                     paddingStyle:MPSGraphPaddingStyleExplicit
                       dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                    weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

  MPSGraphTensor *convT = [graph convolution2DWithSourceTensor:inT
                                                 weightsTensor:wT
                                                    descriptor:convDesc
                                                          name:@"conv2d"];

  /* Optional bias: broadcast-add [1, C_out, 1, 1] over NCHW output. */
  MPSGraphTensor *resultT = convT;
  MPSGraphTensor *biasT   = nil;

  if (bias) {
    biasT   = [graph placeholderWithShape:@[ @1, @(c_out), @1, @1 ]
                                 dataType:MPSDataTypeFloat32
                                     name:@"bias"];
    resultT = [graph additionWithPrimaryTensor:convT
                               secondaryTensor:biasT
                                          name:@"add_bias"];
  }

  /* ---- Prepare feed data ---- */
  MPSGraphDevice *gDev = [MPSGraphDevice deviceWithMTLDevice:device];

  const size_t inBytes = n * c_in * h_in * w_in * sizeof(float);
  const size_t wBytes  = c_out * c_in * k_h * k_w * sizeof(float);

  MPSGraphTensorData *inData = [[MPSGraphTensorData alloc]
      initWithDevice:gDev
                data:[NSData dataWithBytes:input length:inBytes]
               shape:inShape
            dataType:MPSDataTypeFloat32];

  MPSGraphTensorData *wData = [[MPSGraphTensorData alloc]
      initWithDevice:gDev
                data:[NSData dataWithBytes:weight length:wBytes]
               shape:wShape
            dataType:MPSDataTypeFloat32];

  NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
      [NSMutableDictionary dictionaryWithCapacity:3];
  feeds[inT] = inData;
  feeds[wT]  = wData;

  if (bias && biasT) {
    const size_t bBytes = c_out * sizeof(float);
    MPSGraphTensorData *bData = [[MPSGraphTensorData alloc]
        initWithDevice:gDev
                  data:[NSData dataWithBytes:bias length:bBytes]
                 shape:@[ @1, @(c_out), @1, @1 ]
              dataType:MPSDataTypeFloat32];
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
  if (!outData) {
    return false;
  }

  /* Copy NCHW result back to caller buffer. */
  [outData.mpsndarray readBytes:output strideBytes:nil];

  return true;

  } // @autoreleasepool
}