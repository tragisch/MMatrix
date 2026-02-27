/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef DMMa_SPARSE_H
#define DMMa_SPARSE_H

#if defined(__has_include)
#if __has_include(<cs.h>)
#include <cs.h>  // SuiteSparse: Common Sparse Matrix Package
#define DMS_HAS_CSPARSE 1
#endif
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifndef MMATRIX_DEPRECATED
#if defined(__GNUC__) || defined(__clang__)
#define MMATRIX_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define MMATRIX_DEPRECATED(msg)
#endif
#endif

// Forward declaration for SuiteSparse type when cs.h is not available.
#ifndef DMS_HAS_CSPARSE
typedef struct cs_di_sparse cs;
#endif

// Sparse COO matrix; duplicates are allowed and summed in dms_get().
typedef struct DoubleSparseMatrix {
  size_t rows;
  size_t cols;
  size_t nnz;
  size_t capacity;
  size_t *row_indices;
  size_t *col_indices;
  double *values;
} DoubleSparseMatrix;

// Create empty sparse matrix metadata (nnz = 0, arrays NULL).
DoubleSparseMatrix *dms_create_empty(void);

// Create matrix from external COO arrays (no copy, caller manages arrays lifetime).
DoubleSparseMatrix *dms_create_with_values(size_t rows, size_t cols, size_t nnz,
                                           size_t *row_indices,
                                           size_t *col_indices, double *values);

// Create empty COO matrix with pre-allocated triple capacity.
DoubleSparseMatrix *dms_create(size_t rows, size_t cols, size_t capacity);

// Create deep copy of sparse matrix.
DoubleSparseMatrix *dms_clone(const DoubleSparseMatrix *m);
MMATRIX_DEPRECATED("Use dms_clone instead")
DoubleSparseMatrix *dms_create_clone(const DoubleSparseMatrix *m);

// Create sparse n x n identity matrix.
DoubleSparseMatrix *dms_create_identity(size_t n);

// Create sparse random matrix using global RNG state (not thread-safe).
DoubleSparseMatrix *dms_create_random(size_t rows, size_t cols, double density);

// Create deterministic sparse random matrix from explicit seed.
DoubleSparseMatrix *dms_create_random_seeded(size_t rows, size_t cols,
                                              double density, uint64_t seed);

// Set global RNG seed used by non-seeded random creators (not thread-safe).
void dms_set_random_seed(uint64_t seed);

// Get current global RNG seed.
uint64_t dms_get_random_seed(void);

// Conversion helpers (COO <-> cs, COO -> dense array).
cs *dms_to_cs(const DoubleSparseMatrix *coo);
DoubleSparseMatrix *dms_from_cs(const cs *A);
MMATRIX_DEPRECATED("Use dms_from_cs instead")
DoubleSparseMatrix *cs_to_dms(const cs *A);
double *dms_to_array(const DoubleSparseMatrix *mat);

// Dense import helpers.
DoubleSparseMatrix *dms_create_from_array(size_t rows, size_t cols,
                                          double *array);
DoubleSparseMatrix *dms_from_array_static(size_t rows, size_t cols,
                                          double array[rows][cols]);
MMATRIX_DEPRECATED("Use dms_from_array_static instead")
DoubleSparseMatrix *dms_create_from_2D_array(size_t rows, size_t cols,
                                             double array[rows][cols]);

// Append COO triple (i, j, value); duplicates are allowed and summed in dms_get().
bool dms_set(DoubleSparseMatrix *mat, size_t i, size_t j, double value);

// Read value at (i, j) as sum of matching COO triples (O(nnz) worst case).
double dms_get(const DoubleSparseMatrix *mat, size_t i, size_t j);

// Return row i as new sparse matrix.
DoubleSparseMatrix *dms_get_row(const DoubleSparseMatrix *mat, size_t i);
// Return last row as new sparse matrix.
DoubleSparseMatrix *dms_get_last_row(const DoubleSparseMatrix *mat);
// Return column j as new sparse matrix.
DoubleSparseMatrix *dms_get_col(const DoubleSparseMatrix *mat, size_t j);
// Return last column as new sparse matrix.
DoubleSparseMatrix *dms_get_last_col(const DoubleSparseMatrix *mat);

// Matrix operations.
DoubleSparseMatrix *dms_multiply(const DoubleSparseMatrix *mat1,
                                 const DoubleSparseMatrix *mat2);
DoubleSparseMatrix *dms_multiply_by_number(const DoubleSparseMatrix *mat,
                                           const double number);
DoubleSparseMatrix *dms_transpose(const DoubleSparseMatrix *mat);

// Matrix properties.
double dms_density(const DoubleSparseMatrix *mat);

// Print sparse matrix to stdout (debug helper).
void dms_print(const DoubleSparseMatrix *mat);

// Reallocate COO arrays to new_capacity (must be >= nnz).
bool dms_realloc(DoubleSparseMatrix *mat, size_t new_capacity);

// Destroy sparse matrix; safe on NULL. Do not call twice on same pointer.
void dms_destroy(DoubleSparseMatrix *mat);

#endif  // DMMa_SPARSE_H
