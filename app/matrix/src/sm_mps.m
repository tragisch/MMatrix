/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#import "sm_mps.h"

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <stdlib.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

@interface SmMpsMatrixHandle : NSObject
@property(nonatomic, assign) size_t rows;
@property(nonatomic, assign) size_t cols;
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, strong) MPSMatrixDescriptor *descriptor;
@property(nonatomic, strong) MPSMatrix *matrix;
@end

@implementation SmMpsMatrixHandle
@end

@interface SmMpsStreamHandle : NSObject
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLCommandBuffer> commandBuffer;
@property(nonatomic, strong) NSMutableArray *pendingBuffers;
@property(nonatomic, assign) BOOL hasEncodedWork;
@end

@implementation SmMpsStreamHandle
@end

@interface SmMpsGemmPlanHandle : NSObject
@property(nonatomic, assign) size_t resultRows;
@property(nonatomic, assign) size_t resultCols;
@property(nonatomic, assign) size_t interiorCols;
@property(nonatomic, assign) BOOL transposeLeft;
@property(nonatomic, assign) BOOL transposeRight;
@property(nonatomic, strong) MPSMatrixMultiplication *kernel;
@end

@implementation SmMpsGemmPlanHandle
@end

struct SmMpsMatrix {
  void *handle;
};

struct SmMpsStream {
  void *handle;
};

struct SmMpsGemmPlan {
  void *handle;
};

static atomic_ullong g_matrix_allocations;
static atomic_ullong g_command_buffers_created;
static atomic_ullong g_commits;
static atomic_ullong g_waits;
static atomic_ullong g_gemm_encodes;
static atomic_ullong g_uploads;
static atomic_ullong g_downloads;
static atomic_ullong g_plan_allocations;

static void sm_mps_counter_inc(atomic_ullong *counter) {
  atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
}

SmMpsCounters sm_mps_get_counters(void) {
  return (SmMpsCounters){
      .matrix_allocations =
          atomic_load_explicit(&g_matrix_allocations, memory_order_relaxed),
      .command_buffers_created =
          atomic_load_explicit(&g_command_buffers_created, memory_order_relaxed),
      .commits = atomic_load_explicit(&g_commits, memory_order_relaxed),
      .waits = atomic_load_explicit(&g_waits, memory_order_relaxed),
      .gemm_encodes = atomic_load_explicit(&g_gemm_encodes, memory_order_relaxed),
      .uploads = atomic_load_explicit(&g_uploads, memory_order_relaxed),
      .downloads = atomic_load_explicit(&g_downloads, memory_order_relaxed),
      .plan_allocations =
          atomic_load_explicit(&g_plan_allocations, memory_order_relaxed),
  };
}

void sm_mps_reset_counters(void) {
  atomic_store_explicit(&g_matrix_allocations, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_command_buffers_created, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_commits, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_waits, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_gemm_encodes, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_uploads, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_downloads, 0u, memory_order_relaxed);
  atomic_store_explicit(&g_plan_allocations, 0u, memory_order_relaxed);
}

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

static id<MTLComputePipelineState> _sm_mps_bias_relu_pipeline(void) {
  static id<MTLComputePipelineState> pipeline = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    id<MTLDevice> device = _mps_shared_device();
    if (!device) {
      return;
    }
    NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void sm_mps_bias_relu(device float *c [[buffer(0)]],\n"
         "                            device const float *bias [[buffer(1)]],\n"
         "                            constant uint &cols [[buffer(2)]],\n"
         "                            constant uint &total [[buffer(3)]],\n"
         "                            constant uint &has_bias [[buffer(4)]],\n"
         "                            constant uint &bias_is_row [[buffer(5)]],\n"
         "                            uint gid [[thread_position_in_grid]]) {\n"
         "  if (gid >= total) return;\n"
         "  float v = c[gid];\n"
         "  if (has_bias != 0) {\n"
         "    uint bias_idx = (bias_is_row != 0) ? (gid % cols) : gid;\n"
         "    v += bias[bias_idx];\n"
         "  }\n"
         "  c[gid] = max(v, 0.0f);\n"
         "}\n";
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&error];
    if (!library) {
      return;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"sm_mps_bias_relu"];
    if (!function) {
      return;
    }
    pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  });
  return pipeline;
}

void *mps_get_shared_device(void) {
  return (__bridge void *)_mps_shared_device();
}

void *mps_get_shared_command_queue(void) {
  return (__bridge void *)_mps_shared_command_queue();
}

static SmMpsMatrixHandle *_sm_mps_matrix_handle(const SmMpsMatrix *matrix) {
  if (!matrix || !matrix->handle) {
    return nil;
  }
  return (__bridge SmMpsMatrixHandle *)matrix->handle;
}

static SmMpsStreamHandle *_sm_mps_stream_handle(const SmMpsStream *stream) {
  if (!stream || !stream->handle) {
    return nil;
  }
  return (__bridge SmMpsStreamHandle *)stream->handle;
}

static SmMpsGemmPlanHandle *_sm_mps_plan_handle(const SmMpsGemmPlan *plan) {
  if (!plan || !plan->handle) {
    return nil;
  }
  return (__bridge SmMpsGemmPlanHandle *)plan->handle;
}

static id<MTLCommandBuffer> _sm_mps_stream_command_buffer(SmMpsStreamHandle *handle) {
  if (!handle) {
    return nil;
  }
  if (!handle.commandBuffer) {
    handle.commandBuffer = [handle.queue commandBuffer];
    if (handle.commandBuffer) {
      sm_mps_counter_inc(&g_command_buffers_created);
    }
  }
  return handle.commandBuffer;
}

SmMpsMatrix *sm_mps_matrix_create(size_t rows, size_t cols) {
  if (rows == 0 || cols == 0) {
    return NULL;
  }

  @autoreleasepool {
    id<MTLDevice> device = _mps_shared_device();
    if (!device) {
      return NULL;
    }

    const size_t bytes = rows * cols * sizeof(float);
    id<MTLBuffer> buffer =
        [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) {
      return NULL;
    }

    MPSMatrixDescriptor *descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                              columns:cols
                                             rowBytes:cols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrix *matrix = [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
    if (!matrix) {
      return NULL;
    }

    SmMpsMatrixHandle *handle = [SmMpsMatrixHandle new];
    handle.rows = rows;
    handle.cols = cols;
    handle.buffer = buffer;
    handle.descriptor = descriptor;
    handle.matrix = matrix;

    SmMpsMatrix *out = (SmMpsMatrix *)calloc(1, sizeof(SmMpsMatrix));
    if (!out) {
      return NULL;
    }
    out->handle = (__bridge_retained void *)handle;
    sm_mps_counter_inc(&g_matrix_allocations);
    return out;
  }
}

void sm_mps_matrix_destroy(SmMpsMatrix *matrix) {
  if (!matrix) {
    return;
  }
  if (matrix->handle) {
    CFBridgingRelease(matrix->handle);
  }
  free(matrix);
}

float *sm_mps_matrix_contents(SmMpsMatrix *matrix) {
  SmMpsMatrixHandle *handle = _sm_mps_matrix_handle(matrix);
  if (!handle) {
    return NULL;
  }
  return (float *)handle.buffer.contents;
}

const float *sm_mps_matrix_const_contents(const SmMpsMatrix *matrix) {
  SmMpsMatrixHandle *handle = _sm_mps_matrix_handle(matrix);
  if (!handle) {
    return NULL;
  }
  return (const float *)handle.buffer.contents;
}

bool sm_mps_matrix_upload(SmMpsMatrix *matrix, const float *values) {
  SmMpsMatrixHandle *handle = _sm_mps_matrix_handle(matrix);
  if (!handle || !values) {
    return false;
  }
  memcpy(handle.buffer.contents, values, handle.rows * handle.cols * sizeof(float));
  sm_mps_counter_inc(&g_uploads);
  return true;
}

bool sm_mps_matrix_download(const SmMpsMatrix *matrix, float *values) {
  SmMpsMatrixHandle *handle = _sm_mps_matrix_handle(matrix);
  if (!handle || !values) {
    return false;
  }
  memcpy(values, handle.buffer.contents, handle.rows * handle.cols * sizeof(float));
  sm_mps_counter_inc(&g_downloads);
  return true;
}

SmMpsStream *sm_mps_stream_create(void) {
  @autoreleasepool {
    id<MTLCommandQueue> queue = _mps_shared_command_queue();
    if (!queue) {
      return NULL;
    }

    SmMpsStreamHandle *handle = [SmMpsStreamHandle new];
    handle.queue = queue;
    handle.pendingBuffers = [NSMutableArray array];

    SmMpsStream *stream = (SmMpsStream *)calloc(1, sizeof(SmMpsStream));
    if (!stream) {
      return NULL;
    }
    stream->handle = (__bridge_retained void *)handle;
    return stream;
  }
}

bool sm_mps_stream_commit(SmMpsStream *stream) {
  SmMpsStreamHandle *handle = _sm_mps_stream_handle(stream);
  if (!handle) {
    return false;
  }
  if (!handle.commandBuffer) {
    return true;
  }
  if (!handle.hasEncodedWork) {
    handle.commandBuffer = nil;
    return true;
  }

  [handle.commandBuffer commit];
  [handle.pendingBuffers addObject:handle.commandBuffer];
  handle.commandBuffer = nil;
  handle.hasEncodedWork = NO;
  sm_mps_counter_inc(&g_commits);
  return true;
}

bool sm_mps_stream_wait(SmMpsStream *stream) {
  SmMpsStreamHandle *handle = _sm_mps_stream_handle(stream);
  if (!handle) {
    return false;
  }
  if (!sm_mps_stream_commit(stream)) {
    return false;
  }

  bool ok = true;
  for (id<MTLCommandBuffer> commandBuffer in handle.pendingBuffers) {
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status == MTLCommandBufferStatusError) {
      ok = false;
    }
  }
  [handle.pendingBuffers removeAllObjects];
  sm_mps_counter_inc(&g_waits);
  return ok;
}

void sm_mps_stream_destroy(SmMpsStream *stream) {
  if (!stream) {
    return;
  }
  if (stream->handle) {
    SmMpsStreamHandle *handle = _sm_mps_stream_handle(stream);
    if (handle && (handle.commandBuffer || handle.pendingBuffers.count > 0)) {
      (void)sm_mps_stream_wait(stream);
    }
    CFBridgingRelease(stream->handle);
  }
  free(stream);
}

SmMpsGemmPlan *sm_mps_gemm_plan_create(size_t result_rows,
                                       size_t result_cols,
                                       size_t interior_cols,
                                       bool transpose_left,
                                       bool transpose_right,
                                       float alpha,
                                       float beta) {
  if (result_rows == 0 || result_cols == 0 || interior_cols == 0) {
    return NULL;
  }

  @autoreleasepool {
    id<MTLDevice> device = _mps_shared_device();
    if (!device) {
      return NULL;
    }

    MPSMatrixMultiplication *kernel =
        [[MPSMatrixMultiplication alloc] initWithDevice:device
                                          transposeLeft:(BOOL)transpose_left
                                         transposeRight:(BOOL)transpose_right
                                             resultRows:result_rows
                                          resultColumns:result_cols
                                        interiorColumns:interior_cols
                                                  alpha:alpha
                                                   beta:beta];
    if (!kernel) {
      return NULL;
    }

    SmMpsGemmPlanHandle *handle = [SmMpsGemmPlanHandle new];
    handle.resultRows = result_rows;
    handle.resultCols = result_cols;
    handle.interiorCols = interior_cols;
    handle.transposeLeft = (BOOL)transpose_left;
    handle.transposeRight = (BOOL)transpose_right;
    handle.kernel = kernel;

    SmMpsGemmPlan *plan = (SmMpsGemmPlan *)calloc(1, sizeof(SmMpsGemmPlan));
    if (!plan) {
      return NULL;
    }
    plan->handle = (__bridge_retained void *)handle;
    sm_mps_counter_inc(&g_plan_allocations);
    return plan;
  }
}

void sm_mps_gemm_plan_destroy(SmMpsGemmPlan *plan) {
  if (!plan) {
    return;
  }
  if (plan->handle) {
    CFBridgingRelease(plan->handle);
  }
  free(plan);
}

static bool _sm_mps_gemm_plan_matches(const SmMpsGemmPlanHandle *plan,
                                      const SmMpsMatrixHandle *handleC,
                                      const SmMpsMatrixHandle *handleA,
                                      const SmMpsMatrixHandle *handleB) {
  if (!plan || !handleA || !handleB || !handleC) {
    return false;
  }
  const size_t left_rows = plan.transposeLeft ? handleA.cols : handleA.rows;
  const size_t left_cols = plan.transposeLeft ? handleA.rows : handleA.cols;
  const size_t right_rows = plan.transposeRight ? handleB.cols : handleB.rows;
  const size_t right_cols = plan.transposeRight ? handleB.rows : handleB.cols;
  return left_cols == right_rows &&
         left_cols == plan.interiorCols &&
         handleC.rows == left_rows &&
         handleC.cols == right_cols &&
         handleC.rows == plan.resultRows &&
         handleC.cols == plan.resultCols;
}

bool sm_mps_gemm_plan_encode(SmMpsStream *stream, const SmMpsGemmPlan *plan,
                             SmMpsMatrix *C, const SmMpsMatrix *A,
                             const SmMpsMatrix *B) {
  SmMpsStreamHandle *streamHandle = _sm_mps_stream_handle(stream);
  SmMpsGemmPlanHandle *planHandle = _sm_mps_plan_handle(plan);
  SmMpsMatrixHandle *handleA = _sm_mps_matrix_handle(A);
  SmMpsMatrixHandle *handleB = _sm_mps_matrix_handle(B);
  SmMpsMatrixHandle *handleC = _sm_mps_matrix_handle(C);
  if (!streamHandle || !planHandle ||
      !_sm_mps_gemm_plan_matches(planHandle, handleC, handleA, handleB)) {
    return false;
  }

  id<MTLCommandBuffer> commandBuffer =
      _sm_mps_stream_command_buffer(streamHandle);
  if (!commandBuffer) {
    return false;
  }

  [planHandle.kernel encodeToCommandBuffer:commandBuffer
                                leftMatrix:handleA.matrix
                               rightMatrix:handleB.matrix
                              resultMatrix:handleC.matrix];
  streamHandle.hasEncodedWork = YES;
  sm_mps_counter_inc(&g_gemm_encodes);
  return true;
}

bool sm_mps_matrix_gemm_async(SmMpsStream *stream, SmMpsMatrix *C, float alpha,
                              const SmMpsMatrix *A, bool transpose_left,
                              const SmMpsMatrix *B, bool transpose_right,
                              float beta) {
  SmMpsMatrixHandle *handleA = _sm_mps_matrix_handle(A);
  SmMpsMatrixHandle *handleB = _sm_mps_matrix_handle(B);
  SmMpsMatrixHandle *handleC = _sm_mps_matrix_handle(C);
  if (!handleA || !handleB || !handleC) {
    return false;
  }

  size_t left_rows = transpose_left ? handleA.cols : handleA.rows;
  size_t left_cols = transpose_left ? handleA.rows : handleA.cols;
  size_t right_rows = transpose_right ? handleB.cols : handleB.rows;
  size_t right_cols = transpose_right ? handleB.rows : handleB.cols;
  if (left_cols != right_rows ||
      handleC.rows != left_rows || handleC.cols != right_cols) {
    return false;
  }

  SmMpsGemmPlan *plan =
      sm_mps_gemm_plan_create(handleC.rows, handleC.cols, left_cols,
                              transpose_left, transpose_right, alpha, beta);
  if (!plan) {
    return false;
  }
  bool ok = sm_mps_gemm_plan_encode(stream, plan, C, A, B);
  sm_mps_gemm_plan_destroy(plan);
  return ok;
}

bool sm_mps_matrix_bias_relu_async(SmMpsStream *stream, SmMpsMatrix *C,
                                   const SmMpsMatrix *bias,
                                   bool bias_is_row) {
  SmMpsStreamHandle *streamHandle = _sm_mps_stream_handle(stream);
  SmMpsMatrixHandle *handleC = _sm_mps_matrix_handle(C);
  SmMpsMatrixHandle *handleBias = _sm_mps_matrix_handle(bias);
  if (!streamHandle || !handleC) {
    return false;
  }

  const bool has_bias = handleBias != nil;
  if (has_bias) {
    if (bias_is_row) {
      if (handleBias.rows != 1 || handleBias.cols != handleC.cols) {
        return false;
      }
    } else if (handleBias.rows != handleC.rows || handleBias.cols != handleC.cols) {
      return false;
    }
  }
  if (handleC.cols > UINT32_MAX ||
      handleC.rows > UINT32_MAX / handleC.cols) {
    return false;
  }

  id<MTLComputePipelineState> pipeline = _sm_mps_bias_relu_pipeline();
  id<MTLCommandBuffer> commandBuffer = _sm_mps_stream_command_buffer(streamHandle);
  if (!pipeline || !commandBuffer) {
    return false;
  }
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder) {
    return false;
  }

  uint32_t cols = (uint32_t)handleC.cols;
  uint32_t total = (uint32_t)(handleC.rows * handleC.cols);
  uint32_t hasBias = has_bias ? 1u : 0u;
  uint32_t biasIsRow = bias_is_row ? 1u : 0u;

  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:handleC.buffer offset:0 atIndex:0];
  [encoder setBuffer:(has_bias ? handleBias.buffer : handleC.buffer)
              offset:0
             atIndex:1];
  [encoder setBytes:&cols length:sizeof(cols) atIndex:2];
  [encoder setBytes:&total length:sizeof(total) atIndex:3];
  [encoder setBytes:&hasBias length:sizeof(hasBias) atIndex:4];
  [encoder setBytes:&biasIsRow length:sizeof(biasIsRow) atIndex:5];

  NSUInteger width = pipeline.threadExecutionWidth;
  if (width == 0) {
    width = 64;
  }
  MTLSize threadsPerThreadgroup = MTLSizeMake(width, 1, 1);
  MTLSize threadsPerGrid = MTLSizeMake(total, 1, 1);
  [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
  [encoder endEncoding];
  streamHandle.hasEncodedWork = YES;
  return true;
}

bool sm_mps_matrix_gemm_bias_relu_async(SmMpsStream *stream, SmMpsMatrix *C,
                                        float alpha, const SmMpsMatrix *A,
                                        bool transpose_left,
                                        const SmMpsMatrix *B,
                                        bool transpose_right,
                                        const SmMpsMatrix *bias,
                                        bool bias_is_row) {
  return sm_mps_matrix_gemm_async(stream, C, alpha, A, transpose_left, B,
                                  transpose_right, 0.0f) &&
         sm_mps_matrix_bias_relu_async(stream, C, bias, bias_is_row);
}

bool sm_mps_matrix_gemm_bias_relu_ex(SmMpsMatrix *C, float alpha,
                                     const SmMpsMatrix *A,
                                     bool transpose_left,
                                     const SmMpsMatrix *B,
                                     bool transpose_right,
                                     const SmMpsMatrix *bias,
                                     bool bias_is_row) {
  SmMpsStream *stream = sm_mps_stream_create();
  if (!stream) {
    return false;
  }
  bool ok = sm_mps_matrix_gemm_bias_relu_async(stream, C, alpha, A,
                                               transpose_left, B,
                                               transpose_right, bias,
                                               bias_is_row);
  if (ok) {
    ok = sm_mps_stream_wait(stream);
  }
  sm_mps_stream_destroy(stream);
  return ok;
}

bool sm_mps_matrix_gemm_ex(SmMpsMatrix *C, float alpha,
                           const SmMpsMatrix *A, bool transpose_left,
                           const SmMpsMatrix *B, bool transpose_right,
                           float beta) {
  SmMpsStream *stream = sm_mps_stream_create();
  if (!stream) {
    return false;
  }
  bool ok = sm_mps_matrix_gemm_async(stream, C, alpha, A, transpose_left,
                                     B, transpose_right, beta);
  if (ok) {
    ok = sm_mps_stream_wait(stream);
  }
  sm_mps_stream_destroy(stream);
  return ok;
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
  sm_mps_counter_inc(&g_uploads);
  sm_mps_counter_inc(&g_uploads);
  id<MTLBuffer> bufferC = nil;
  size_t c_bytes = result_rows * result_cols * sizeof(float);
  if (beta != 0.0f) {
    bufferC = [device newBufferWithBytes:result
                                  length:c_bytes
                                 options:MTLResourceStorageModeShared];
    sm_mps_counter_inc(&g_uploads);
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
  sm_mps_counter_inc(&g_command_buffers_created);

  [mm encodeToCommandBuffer:commandBuffer
                 leftMatrix:matrixA
                rightMatrix:matrixB
               resultMatrix:matrixC];
  sm_mps_counter_inc(&g_gemm_encodes);
  [commandBuffer commit];
  sm_mps_counter_inc(&g_commits);
  /* This one-shot API returns a host-visible result, so this is a true CPU boundary. */
  [commandBuffer waitUntilCompleted];
  sm_mps_counter_inc(&g_waits);

  if (commandBuffer.status == MTLCommandBufferStatusError) {
    return false;
  }

  memcpy(result, matrixC.data.contents, c_bytes);
  sm_mps_counter_inc(&g_downloads);
  return true;

  } // @autoreleasepool
}

bool mps_matrix_multiply(const float *mat1, size_t rows1, size_t cols1,
                         const float *mat2, size_t rows2, size_t cols2,
                         float *result) {
  return mps_matrix_multiply_ex(mat1, rows1, cols1, false, mat2, rows2, cols2,
                                false, 1.0f, 0.0f, result, rows1, cols2);
}
