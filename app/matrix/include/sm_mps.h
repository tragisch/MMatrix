/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef SM_MPS_H
#define SM_MPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

/*
 * Explicit Apple Silicon MPS helpers. This API is separate from sm.h because
 * callers should opt into GPU boundary costs and keep data GPU-resident when
 * they need real throughput.
 */

/// Returns the shared MTLDevice as an opaque pointer, or NULL if unavailable.
void *mps_get_shared_device(void);

/// Returns the shared MTLCommandQueue as an opaque pointer, or NULL if unavailable.
void *mps_get_shared_command_queue(void);

typedef struct SmMpsMatrix SmMpsMatrix;
typedef struct SmMpsStream SmMpsStream;
typedef struct SmMpsGemmPlan SmMpsGemmPlan;

typedef struct SmMpsCounters {
  unsigned long long matrix_allocations;
  unsigned long long command_buffers_created;
  unsigned long long commits;
  unsigned long long waits;
  unsigned long long gemm_encodes;
  unsigned long long uploads;
  unsigned long long downloads;
  unsigned long long plan_allocations;
} SmMpsCounters;

SmMpsCounters sm_mps_get_counters(void);
void sm_mps_reset_counters(void);

SmMpsMatrix *sm_mps_matrix_create(size_t rows, size_t cols);
void sm_mps_matrix_destroy(SmMpsMatrix *matrix);
bool sm_mps_matrix_upload(SmMpsMatrix *matrix, const float *values);
bool sm_mps_matrix_download(const SmMpsMatrix *matrix, float *values);

SmMpsStream *sm_mps_stream_create(void);
bool sm_mps_stream_commit(SmMpsStream *stream);
bool sm_mps_stream_wait(SmMpsStream *stream);
void sm_mps_stream_destroy(SmMpsStream *stream);

SmMpsGemmPlan *sm_mps_gemm_plan_create(size_t result_rows,
                                       size_t result_cols,
                                       size_t interior_cols,
                                       bool transpose_left,
                                       bool transpose_right,
                                       float alpha,
                                       float beta);
void sm_mps_gemm_plan_destroy(SmMpsGemmPlan *plan);
bool sm_mps_gemm_plan_encode(SmMpsStream *stream, const SmMpsGemmPlan *plan,
                             SmMpsMatrix *C, const SmMpsMatrix *A,
                             const SmMpsMatrix *B);

bool sm_mps_matrix_gemm_async(SmMpsStream *stream, SmMpsMatrix *C, float alpha,
                              const SmMpsMatrix *A, bool transpose_left,
                              const SmMpsMatrix *B, bool transpose_right,
                              float beta);

bool sm_mps_matrix_gemm_ex(SmMpsMatrix *C, float alpha,
                           const SmMpsMatrix *A, bool transpose_left,
                           const SmMpsMatrix *B, bool transpose_right,
                           float beta);

bool mps_matrix_multiply(const float *mat1, size_t rows1, size_t cols1,
                         const float *mat2, size_t rows2, size_t cols2,
                         float *result);

bool mps_matrix_multiply_ex(const float *mat1, size_t rows1, size_t cols1,
                            bool transpose_left, const float *mat2,
                            size_t rows2, size_t cols2, bool transpose_right,
                            float alpha, float beta, float *result,
                            size_t result_rows, size_t result_cols);

#ifdef __cplusplus
}
#endif

#endif
