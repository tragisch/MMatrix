/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#import "sm_mps.h"

void mps_matrix_multiply(const float *mat1, size_t rows1, size_t cols1,
                         const float *mat2, size_t rows2, size_t cols2,
                         float *result) {
  if (cols1 != rows2) {
    NSLog(@"Matrix dimensions do not match for multiplication.");
    return;
  }

  // Metal device
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    NSLog(@"Metal is not supported on this device.");
    return;
  }

  // Command queue
  id<MTLCommandQueue> commandQueue = [device newCommandQueue];

  // Create MPSMatrixDescriptors
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
      [MPSMatrixDescriptor matrixDescriptorWithRows:rows1
                                            columns:cols2
                                           rowBytes:cols2 * sizeof(float)
                                           dataType:MPSDataTypeFloat32];

  // Create MPSMatrices
  MPSMatrix *matrixA = [[MPSMatrix alloc]
      initWithBuffer:[device newBufferWithBytes:mat1
                                         length:rows1 * cols1 * sizeof(float)
                                        options:MTLResourceStorageModeShared]
          descriptor:descA];
  MPSMatrix *matrixB = [[MPSMatrix alloc]
      initWithBuffer:[device newBufferWithBytes:mat2
                                         length:rows2 * cols2 * sizeof(float)
                                        options:MTLResourceStorageModeShared]
          descriptor:descB];
  MPSMatrix *matrixC = [[MPSMatrix alloc]
      initWithBuffer:[device newBufferWithLength:rows1 * cols2 * sizeof(float)
                                         options:MTLResourceStorageModeShared]
          descriptor:descC];

  // Create MPSMatrixMultiplication
  MPSMatrixMultiplication *matrixMultiplication =
      [[MPSMatrixMultiplication alloc] initWithDevice:device
                                        transposeLeft:NO
                                       transposeRight:NO
                                           resultRows:rows1
                                        resultColumns:cols2
                                      interiorColumns:cols1
                                                alpha:1.0
                                                 beta:0.0];

  // Encode command
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  [matrixMultiplication encodeToCommandBuffer:commandBuffer
                                   leftMatrix:matrixA
                                  rightMatrix:matrixB
                                 resultMatrix:matrixC];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  // Copy result back to CPU
  memcpy(result, matrixC.data.contents, rows1 * cols2 * sizeof(float));
}
//