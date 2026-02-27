/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef DM_H
#define DM_H

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
/*         Double Matrix Type         */
/**************************************/

// Dense double matrix with row-major storage: values[i * cols + j].
typedef struct DoubleMatrix {
  size_t rows;
  size_t cols;
  size_t capacity;
  double *values;
} DoubleMatrix;

/**************************************/
/*         Matrix Creation            */
/**************************************/

// Create empty matrix metadata (values == NULL), returns NULL on allocation failure.
DoubleMatrix *dm_create_empty(void);

// Create matrix from external values pointer (no copy, caller manages values lifetime).
DoubleMatrix *dm_create_with_values(size_t rows, size_t cols, double *values);

// Create uninitialized matrix (rows x cols), returns NULL on allocation failure.
DoubleMatrix *dm_create(size_t rows, size_t cols);

// Create deep copy of matrix, returns NULL on allocation failure.
DoubleMatrix *dm_clone(const DoubleMatrix *m);
MMATRIX_DEPRECATED("Use dm_clone instead")
DoubleMatrix *dm_create_clone(const DoubleMatrix *m);

// Create n x n identity matrix, returns NULL on allocation failure.
DoubleMatrix *dm_create_identity(size_t n);

// Create random matrix using global RNG state (not thread-safe).
DoubleMatrix *dm_create_random(size_t rows, size_t cols);

// Create deterministic random matrix from explicit seed.
DoubleMatrix *dm_create_random_seeded(size_t rows, size_t cols, uint64_t seed);

// Set global RNG seed used by non-seeded random creators (not thread-safe).
void dm_set_random_seed(uint64_t seed);

// Get current global RNG seed.
uint64_t dm_get_random_seed(void);

/**************************************/
/*         Matrix Import              */
/**************************************/
DoubleMatrix *dm_from_array_ptrs(size_t rows, size_t cols, double **array);
DoubleMatrix *dm_from_array_static(size_t rows, size_t cols,
                                   double array[rows][cols]);
MMATRIX_DEPRECATED("Use dm_from_array_ptrs instead")
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols, double **array);
MMATRIX_DEPRECATED("Use dm_from_array_static instead")
DoubleMatrix *dm_create_from_2D_array(size_t rows, size_t cols,
                                      double array[rows][cols]);

/**************************************/
/*         Matrix Accessors           */
/**************************************/

// Read element at (i, j); caller must ensure bounds are valid.
double dm_get(const DoubleMatrix *mat, size_t i, size_t j);

// Write element at (i, j); concurrent writes to same matrix are not thread-safe.
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value);
// Return row i as new matrix.
DoubleMatrix *dm_get_row(const DoubleMatrix *mat, size_t i);
// Return last row as new matrix.
DoubleMatrix *dm_get_last_row(const DoubleMatrix *mat);
// Return column j as new matrix.
DoubleMatrix *dm_get_col(const DoubleMatrix *mat, size_t j);
// Return last column as new matrix.
DoubleMatrix *dm_get_last_col(const DoubleMatrix *mat);

/**************************************/
/*         Matrix Shape Ops           */
/**************************************/
// Reshape matrix metadata; element count must remain compatible.
void dm_reshape(DoubleMatrix *matrix, size_t new_rows, size_t new_cols);
// Resize matrix storage to new shape.
void dm_resize(DoubleMatrix *mat, size_t new_row, size_t new_col);

/**************************************/
/*       Matrix Arithmetic            */
/**************************************/
DoubleMatrix *dm_multiply(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
DoubleMatrix *dm_multiply_by_number(const DoubleMatrix *mat,
                                    const double number);
DoubleMatrix *dm_elementwise_multiply(const DoubleMatrix *mat1,
                                      const DoubleMatrix *mat2);
DoubleMatrix *dm_div(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
DoubleMatrix *dm_add(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
DoubleMatrix *dm_diff(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
DoubleMatrix *dm_inverse(const DoubleMatrix *mat);

/**************************************/
/*       Matrix Transformations       */
/**************************************/
DoubleMatrix *dm_transpose(const DoubleMatrix *mat);

/**************************************/
/*        In-place Operations         */
/**************************************/
bool dm_inplace_add(DoubleMatrix *mat1, const DoubleMatrix *mat2);
bool dm_inplace_diff(DoubleMatrix *mat1, const DoubleMatrix *mat2);
bool dm_inplace_transpose(DoubleMatrix *mat);
bool dm_inplace_multiply_by_number(DoubleMatrix *mat, const double scalar);
bool dm_inplace_gauss_elimination(DoubleMatrix *mat);
bool dm_inplace_elementwise_multiply(DoubleMatrix *mat1,
                                     const DoubleMatrix *mat2);
bool dm_inplace_div(DoubleMatrix *mat1, const DoubleMatrix *mat2);

/**************************************/
/*         Matrix Properties          */
/**************************************/
double dm_determinant(const DoubleMatrix *mat);
double dm_trace(const DoubleMatrix *mat);
size_t dm_rank(const DoubleMatrix *mat);
double dm_norm(const DoubleMatrix *mat);
double dm_density(const DoubleMatrix *mat);

/**************************************/
/*     Matrix Property Checks         */
/**************************************/
bool dm_is_empty(const DoubleMatrix *mat);
bool dm_is_square(const DoubleMatrix *mat);
bool dm_is_vector(const DoubleMatrix *mat);
bool dm_is_equal_size(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
bool dm_is_equal(const DoubleMatrix *mat1, const DoubleMatrix *mat2);

/**************************************/
/*         Matrix Utilities           */
/**************************************/
// Export matrix as newly allocated column-major array (caller must free()).
double *dm_to_column_major(const DoubleMatrix *mat);

/**************************************/
/*              File I/O              */
/**************************************/
// Print matrix to stdout (debug helper).
void dm_print(const DoubleMatrix *matrix);
// Return active compute backend name.
const char *dm_active_library(void);

/**************************************/
/*         Memory Management          */
/**************************************/

// Destroy matrix; safe on NULL. Do not call twice on the same pointer.
void dm_destroy(DoubleMatrix *mat);

#endif  // DM_H
