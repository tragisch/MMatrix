/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#import "sm_mps.h"

static id<MTLDevice> mps_shared_device(void) {
  static id<MTLDevice> device = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    device = MTLCreateSystemDefaultDevice();
  });
  return device;
}

static id<MTLCommandQueue> mps_shared_command_queue(id<MTLDevice> device) {
  static id<MTLCommandQueue> queue = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    queue = [device newCommandQueue];
  });
  return queue;
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

  id<MTLDevice> device = mps_shared_device();
  if (!device) {
    return false;
  }

  id<MTLCommandQueue> commandQueue = mps_shared_command_queue(device);
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

  memcpy(result, matrixC.data.contents, c_bytes);
  return true;
}

void mps_matrix_multiply(const float *mat1, size_t rows1, size_t cols1,
                         const float *mat2, size_t rows2, size_t cols2,
                         float *result) {
  (void)mps_matrix_multiply_ex(mat1, rows1, cols1, false, mat2, rows2, cols2,
                               false, 1.0f, 0.0f, result, rows1, cols2);
}
//