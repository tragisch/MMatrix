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

// Forward declaration for SuiteSparse type to avoid hard header dependency
// in public API headers (helps IDE/clangd when external include paths differ).
#ifndef DMS_HAS_CSPARSE
typedef struct cs_di_sparse cs;
#endif

/**
 * @struct DoubleSparseMatrix
 * @brief Sparse matrix in Coordinate (COO) format with double values.
 *
 * TENSOR SEMANTICS:
 *   - Storage Format: Coordinate (COO) — list of (row, col, value) triples
 *   - Order: UNSORTED (elements may appear in any order)
 *   - Duplicates: Allowed; duplicate (row, col) pairs are SUMMED on access
 *   - Capacity: Allocated space for triples; nnz ≤ capacity always
 *   - Memory: Not row-major; suitable for sparse matrices (nnz << rows*cols)
 *
 * INDEX FORMULA (unlike dense matrices):
 *   Element at (i, j) = sum of all values[k] where row_indices[k]==i && col_indices[k]==j
 *   Access is O(nnz) in worst case (linear search through all triples)
 *
 * TYPICAL USE CASES:
 *   - Large sparse linear systems (Ax=b)
 *   - Adjacency matrices for graphs/networks
 *   - Document-term matrices (NLP)
 *   - Ratings/recommendations (very sparse user-item matrices)
 *
 * NOT RECOMMENDED FOR:
 *   - Dense data (use DoubleMatrix instead)
 *   - Random access patterns (use CSR/CSC format for dm_get efficiency)
 *   - In-place modifications (append-only design)
 *
 * OWNERSHIP & MEMORY:
 *   Caller owns struct + all three arrays (row_indices, col_indices, values)
 *   Must deallocate using dms_destroy() or: free(all 3 arrays); free(struct)
 *
 * THREAD-SAFETY:
 *   - dms_get() iterates O(nnz); no global state affected
 *   - dms_set() appends new triple; NOT thread-safe (modifies nnz)
 *   - Random seed functions: NOT thread-safe (global state)
 *
 * @see DoubleMatrix (dm.h) for contrast
 */
// struct of DoubleMatrix
typedef struct DoubleSparseMatrix {
  size_t rows;  /**< Number of rows. */
  size_t cols;  /**< Number of columns. */
  size_t nnz;  /**< Number of explicit non-zero triples stored (nnz ≤ capacity). */
  size_t capacity;  /**< Allocated space for triples (may grow via realloc). */
  size_t *row_indices;  /**< Array of row indices [0..nnz-1], order undefined. */
  size_t *col_indices;  /**< Array of column indices [0..nnz-1], order undefined. */
  double *values;  /**< Array of non-zero values [0..nnz-1]. */
} DoubleSparseMatrix;
/**
 * @brief Create an empty sparse matrix (no triples).
 *
 * SEMANTICS:
 *   - Allocates only the struct, not the triple arrays
 *   - nnz = 0, capacity = 0
 *   - Useful for lazy initialization
 *
 * @return Pointer to DoubleSparseMatrix, or NULL if malloc fails.
 *
 * @see dms_create, dms_destroy
 */
DoubleSparseMatrix *dms_create_empty(void);

/**
 * @brief Create sparse matrix from existing triple arrays (stores pointers).
 *
 * ⚠️ OWNERSHIP WARNING (like dm_create_with_values):
 *   Function stores POINTERS to arrays; does NOT copy.
 *   Caller retains ownership of triple arrays.
 *   Caller must NOT free arrays until matrix is destroyed.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param nnz Number of non-zero triples.
 * @param row_indices Pointer to array of row indices (size ≥ nnz).
 * @param col_indices Pointer to array of column indices (size ≥ nnz).
 * @param values Pointer to array of values (size ≥ nnz).
 *
 * @return DoubleSparseMatrix with external arrays, or NULL if struct malloc fails.
 *
 * @see dms_destroy
 */
DoubleSparseMatrix *dms_create_with_values(size_t rows, size_t cols, size_t nnz,
                                           size_t *row_indices,
                                           size_t *col_indices, double *values);

/**
 * @brief Create empty sparse matrix with pre-allocated capacity.
 *
 * SEMANTICS:
 *   - Allocates struct + triple arrays of given capacity
 *   - nnz starts at 0
 *   - Caller can add up to capacity triples via dms_set()
 *   - When nnz reaches capacity, dms_set() will realloc (double capacity)
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param capacity Pre-allocated space for triples.
 *
 * @return DoubleSparseMatrix with allocated but empty triples, or NULL if malloc fails.
 *
 * @see dms_set, dms_destroy
 */
DoubleSparseMatrix *dms_create(size_t rows, size_t cols, size_t capacity);

/**
 * @brief Create deep copy of sparse matrix.
 *
 * @param m Source matrix (const).
 *
 * @return Independent copy with copied triple arrays, or NULL if allocation failed.
 *
 * @see dms_destroy
 */
DoubleSparseMatrix *dms_create_clone(const DoubleSparseMatrix *m);

/**
 * @brief Create sparse identity matrix.
 *
 * SEMANTICS:
 *   - n×n matrix with 1.0 on diagonal, 0.0 elsewhere
 *   - nnz = n (only diagonal elements stored)
 *   - Capacity = n (may grow if modified)
 *
 * @param n Size of identity matrix (n×n).
 *
 * @return Sparse identity matrix, or NULL if allocation failed.
 *
 * @see dms_destroy
 */
DoubleSparseMatrix *dms_create_identity(size_t n);

/**
 * @brief Create sparse random matrix with given density.
 *
 * SEMANTICS:
 *   - Density = nnz / (rows*cols), typically 0.0 < density < 1.0
 *   - Uses global random seed (set via dms_set_random_seed)
 *   - Non-deterministic by default
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (uses global random seed)
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param density Sparsity level: 0.0 (empty) to 1.0 (dense).
 *
 * @return Sparse random matrix, or NULL if allocation failed.
 *
 * @see dms_create_random_seeded, dms_set_random_seed, dms_density
 */
DoubleSparseMatrix *dms_create_random(size_t rows, size_t cols, double density);

/**
 * @brief Create sparse random matrix with explicit seed (deterministic, thread-safe).
 *
 * SEMANTICS:
 *   - Same seed → Same sparsity pattern and values (reproducible)
 *   - Uses SplitMix64 mixing like dm_create_random_seeded
 *   - Does NOT modify global random seed
 *
 * THREAD-SAFETY:
 *   Thread-safe (no global state modified)
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param density Sparsity level (0.0 to 1.0).
 * @param seed Seed value. Same seed → same output.
 *
 * @return Sparse random matrix, or NULL if allocation failed.
 *
 * @see dms_set_random_seed, dms_create_random
 */
DoubleSparseMatrix *dms_create_random_seeded(size_t rows, size_t cols,
                                              double density, uint64_t seed);

/**
 * @brief Set global random seed for dms_create_random().
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (modifies global state)
 *
 * @param seed Seed value.
 *
 * @see dms_get_random_seed, dms_create_random, dms_create_random_seeded
 */
void dms_set_random_seed(uint64_t seed);

/**
 * @brief Get current global random seed.
 *
 * @return Current seed value.
 *
 * @see dms_set_random_seed
 */
uint64_t dms_get_random_seed(void);

// Converting to cs-sparse format or array
cs *dms_to_cs(const DoubleSparseMatrix *coo);
DoubleSparseMatrix *cs_to_dms(const cs *A);
double *dms_to_array(const DoubleSparseMatrix *mat);

// Importing from array
DoubleSparseMatrix *dms_create_from_array(size_t rows, size_t cols,
                                          double *array);
DoubleSparseMatrix *dms_create_from_2D_array(size_t rows, size_t cols,
                                             double array[rows][cols]);

// Getters and Setters

/**
 * @brief Set element at (i, j) in sparse matrix.
 *
 * SEMANTICS:
 *   - Appends new (i, j, value) triple to matrix
 *   - Does NOT replace existing (i, j) entry; appends another one
 *   - If (i, j) entry already exists, both triples will be summed on dms_get()
 *   - If nnz reaches capacity, arrays are reallocated (capacity doubled)
 *
 * CAPACITY GROWTH:
 *   @code
 *     DoubleSparseMatrix *m = dms_create(10, 10, 5);  // capacity=5
 *     dms_set(m, 0, 0, 1.0);  // nnz=1
 *     dms_set(m, 1, 1, 2.0);  // nnz=2
 *     dms_set(m, 2, 2, 3.0);  // nnz=3
 *     dms_set(m, 3, 3, 4.0);  // nnz=4
 *     dms_set(m, 4, 4, 5.0);  // nnz=5 (at capacity)
 *     dms_set(m, 5, 5, 6.0);  // nnz=6, capacity→10 (reallocated)
 *   @endcode
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (modifies nnz and possibly reallocates)
 *
 * @param mat Matrix (modified).
 * @param i Row index.
 * @param j Column index.
 * @param value Value to add to (i, j).
 *
 * @see dms_get
 */
void dms_set(DoubleSparseMatrix *mat, size_t i, size_t j, double value);

/**
 * @brief Get element at (i, j) in sparse matrix.
 *
 * SEMANTICS:
 *   - Returns sum of all triples with row==i && col==j
 *   - O(nnz) worst case (linear search through all triples)
 *   - Returns 0.0 if no triple exists for (i, j)
 *
 * DUPLICATE HANDLING:
 *   If (i, j) appears multiple times in triples, all are summed:
 *   @code
 *     dms_set(m, 0, 0, 1.0);  // Triple: (0, 0, 1.0)
 *     dms_set(m, 0, 0, 2.0);  // Triple: (0, 0, 2.0)
 *     dms_get(m, 0, 0);       // Returns 3.0 (1.0 + 2.0)
 *   @endcode
 *
 * @param mat Matrix (const).
 * @param i Row index.
 * @param j Column index.
 *
 * @return Sum of all values at (i, j), or 0.0 if not present.
 *
 * @see dms_set
 */
double dms_get(const DoubleSparseMatrix *mat, size_t i, size_t j);

DoubleSparseMatrix *dms_get_row(const DoubleSparseMatrix *mat, size_t i);
DoubleSparseMatrix *dms_get_last_row(const DoubleSparseMatrix *mat);
DoubleSparseMatrix *dms_get_col(const DoubleSparseMatrix *mat, size_t j);
DoubleSparseMatrix *dms_get_last_col(const DoubleSparseMatrix *mat);

// Matrix operations
DoubleSparseMatrix *dms_multiply(const DoubleSparseMatrix *mat1,
                                 const DoubleSparseMatrix *mat2);
DoubleSparseMatrix *dms_multiply_by_number(const DoubleSparseMatrix *mat,
                                           const double number);
DoubleSparseMatrix *dms_transpose(const DoubleSparseMatrix *mat);

// Matrix properties
double dms_density(const DoubleSparseMatrix *mat);

// Matrix properties (boolean)

// In-place operations:

// File I/O
/**
 * @brief Print sparse matrix to stdout (for debugging).
 *
 * @param mat Matrix to print (const).
 *
 * @see dms_write_matrix_market
 */
void dms_print(const DoubleSparseMatrix *mat);

// Memory management

/**
 * @brief Reallocate sparse matrix triple arrays to new capacity.
 *
 * SEMANTICS:
 *   - Resizes capacity while preserving nnz
 *   - Typically called internally when nnz reaches capacity
 *   - Can also be called to trim excess capacity or pre-allocate
 *
 * @param mat Matrix (modified).
 * @param new_capacity New capacity (must be ≥ nnz, or undefined behavior).
 *
 * @see dms_set
 */
void dms_realloc(DoubleSparseMatrix *mat, size_t new_capacity);

/**
 * @brief Free sparse matrix and all associated memory.
 *
 * SEMANTICS:
 *   - Deallocates struct + all three triple arrays
 *   - When called on matrix from dms_create_with_values(),
 *     frees struct only (caller owns the triple arrays)
 *   - Safe to call on NULL (no-op)
 *
 * ⚠️ DO NOT call twice on same pointer (double-free crash).
 *
 * @param mat Matrix to deallocate.
 *
 * @see dms_create, dms_create_empty
 */
void dms_destroy(DoubleSparseMatrix *mat);

#endif  // DMMa_SPARSE_H
