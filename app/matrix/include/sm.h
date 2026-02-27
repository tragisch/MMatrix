/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef SM_H
#define SM_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef MMATRIX_DEPRECATED
#if defined(__GNUC__) || defined(__clang__)
#define MMATRIX_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define MMATRIX_DEPRECATED(msg)
#endif
#endif

/**************************************/
/*         Float Matrix Type          */
/**************************************/

// Dense float matrix with row-major storage: values[i * cols + j].
typedef struct FloatMatrix {
  size_t rows;
  size_t cols;
  size_t capacity;
  float *values;
} FloatMatrix;

/**************************************/
/*         Matrix Creation            */
/**************************************/

// Create empty matrix metadata (values == NULL), returns NULL on allocation failure.
FloatMatrix *sm_create_empty(void);

// Create zero-initialized matrix (rows x cols), returns NULL on allocation failure.
FloatMatrix *sm_create_zeros(size_t rows, size_t cols);

// Create uninitialized matrix (rows x cols), returns NULL on allocation failure.
FloatMatrix *sm_create(size_t rows, size_t cols);

// Create matrix from external values pointer (no copy, caller manages values lifetime).
FloatMatrix *sm_create_with_values(size_t rows, size_t cols, float *values);

// Create deep copy of matrix, returns NULL on allocation failure.
FloatMatrix *sm_clone(const FloatMatrix *m);

// Create n x n identity matrix, returns NULL on allocation failure.
FloatMatrix *sm_create_identity(size_t n);

// Create random matrix using global RNG state (not thread-safe).
FloatMatrix *sm_create_random(size_t rows, size_t cols);

// Create deterministic random matrix from explicit seed.
FloatMatrix *sm_create_random_seeded(size_t rows, size_t cols, uint64_t seed);

// Set global RNG seed used by non-seeded random creators (not thread-safe).
void sm_set_random_seed(uint64_t seed);

// Get current global RNG seed.
uint64_t sm_get_random_seed(void);

// Create He-initialized weights (recommended for ReLU-like networks).
FloatMatrix *sm_create_random_he(size_t rows, size_t cols, size_t fan_in);

// Create Xavier/Glorot-initialized weights (recommended for tanh/sigmoid).
FloatMatrix *sm_create_random_xavier(size_t rows, size_t cols, size_t fan_in,
                                     size_t fan_out);

// Create matrix from row-pointer array (data copied into row-major storage).
FloatMatrix *sm_from_array_ptrs(size_t rows, size_t cols, float **array);

// Create matrix from static 2D array (C99 VLA), data copied.
FloatMatrix *sm_from_array_static(size_t rows, size_t cols,
                                  float array[rows][cols]);

// Export matrix to newly allocated row-major array (caller must free()).
float *sm_to_array(FloatMatrix *matrix);
MMATRIX_DEPRECATED("Use sm_to_array instead")
float *sm_create_array_from_matrix(FloatMatrix *matrix);

/**************************************/
/*         Matrix Accessors           */
/**************************************/

// Read element at (i, j); caller must ensure bounds are valid.
float sm_get(const FloatMatrix *mat, size_t i, size_t j);

// Write element at (i, j); concurrent writes to same matrix are not thread-safe.
void sm_set(FloatMatrix *mat, size_t i, size_t j, float value);
// Return row i as new matrix.
FloatMatrix *sm_get_row(const FloatMatrix *mat, size_t i);
// Return last row as new matrix.
FloatMatrix *sm_get_last_row(const FloatMatrix *mat);
// Return column j as new matrix.
FloatMatrix *sm_get_col(const FloatMatrix *mat, size_t j);
// Return last column as new matrix.
FloatMatrix *sm_get_last_col(const FloatMatrix *mat);
// Return row slice [start, end) as new matrix.
FloatMatrix *sm_slice_rows(const FloatMatrix *mat, size_t start, size_t end);

/**************************************/
/*         Matrix Shape Ops           */
/**************************************/
// Reshape matrix metadata; element count must remain compatible.
void sm_reshape(FloatMatrix *matrix, size_t new_rows, size_t new_cols);
// Resize matrix storage to new shape.
void sm_resize(FloatMatrix *mat, size_t new_row, size_t new_col);

/**************************************/
/*       Matrix Transformations       */
/**************************************/
// Return transposed copy of matrix.
FloatMatrix *sm_transpose(const FloatMatrix *mat);

/**************************************/
/*       Matrix Arithmetic            */
/**************************************/

FloatMatrix *sm_add(const FloatMatrix *mat1, const FloatMatrix *mat2);
FloatMatrix *sm_diff(const FloatMatrix *mat1, const FloatMatrix *mat2);
FloatMatrix *sm_multiply(const FloatMatrix *mat1, const FloatMatrix *mat2);
FloatMatrix *sm_multiply_4(const FloatMatrix *A, const FloatMatrix *B);
FloatMatrix *sm_elementwise_multiply(const FloatMatrix *mat1,
                                     const FloatMatrix *mat2);
FloatMatrix *sm_multiply_by_number(const FloatMatrix *mat, const float number);
FloatMatrix *sm_inverse(const FloatMatrix *mat);
FloatMatrix *sm_div(const FloatMatrix *mat1, const FloatMatrix *mat2);
FloatMatrix *sm_solve_system(const FloatMatrix *A, const FloatMatrix *b);

/**************************************/
/*        Advanced Operations         */
/**************************************/
typedef enum SmTranspose {
  SM_NO_TRANSPOSE = 0,
  SM_TRANSPOSE = 1,
} SmTranspose;

// Advanced BLAS-style kernel:
// C = alpha * op(A) * op(B) + beta * C
// where op(X) is X or X^T depending on transpose flags.
bool sm_gemm(FloatMatrix *C, float alpha, const FloatMatrix *A,
             SmTranspose trans_a, const FloatMatrix *B, SmTranspose trans_b,
             float beta);

// Advanced fused kernel: GEMM + optional bias + ReLU activation.
// bias shape must be 1xcols or rowsxcols.
bool sm_gemm_bias_relu(FloatMatrix *C, const FloatMatrix *A, SmTranspose trans_a,
                       const FloatMatrix *B, SmTranspose trans_b,
                       const FloatMatrix *bias);

/**************************************/
/*        Matrix In-Place Ops         */
/**************************************/
bool sm_inplace_add(FloatMatrix *mat1, const FloatMatrix *mat2);
bool sm_inplace_diff(FloatMatrix *mat1, const FloatMatrix *mat2);
bool sm_inplace_square_transpose(FloatMatrix *mat);
bool sm_inplace_multiply_by_number(FloatMatrix *mat, const float scalar);
bool sm_inplace_elementwise_multiply(FloatMatrix *mat1,
                                     const FloatMatrix *mat2);
bool sm_inplace_div(FloatMatrix *mat1, const FloatMatrix *mat2);
bool sm_inplace_normalize_rows(FloatMatrix *mat);
bool sm_inplace_normalize_cols(FloatMatrix *mat);

/**************************************/
/*       Matrix Properties            */
/**************************************/
float sm_determinant(const FloatMatrix *mat);
float sm_trace(const FloatMatrix *mat);
float sm_norm(const FloatMatrix *mat);
size_t sm_rank(const FloatMatrix *mat);
float sm_density(const FloatMatrix *mat);

/**************************************/
/*       Matrix Property Checks       */
/**************************************/
bool sm_is_empty(const FloatMatrix *mat);
bool sm_is_square(const FloatMatrix *mat);
bool sm_is_vector(const FloatMatrix *mat);
bool sm_is_equal_size(const FloatMatrix *mat1, const FloatMatrix *mat2);
bool sm_is_equal(const FloatMatrix *mat1, const FloatMatrix *mat2);
bool sm_lu_decompose(FloatMatrix *mat, size_t *pivot_order);

/**************************************/
/*         Matrix Utilities           */
/**************************************/

// Print matrix to stdout (debug helper).
void sm_print(const FloatMatrix *matrix);

// Return active compute backend name.
const char *sm_active_library(void);

/**************************************/
/*         Memory Management          */
/**************************************/

// Destroy matrix; safe on NULL. Do not call twice on the same pointer.
void sm_destroy(FloatMatrix *mat);

#endif  // SM_H
