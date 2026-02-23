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
/*         Matrix Struct              */
/**************************************/

/**
 * @struct DoubleMatrix
 * @brief Dense matrix of double values with row-major storage.
 *
 * TENSOR SEMANTICS:
 *   - Storage Format: Row-major (C-contiguous)
 *   - Index Formula: values[i * cols + j] for element at (i, j)
 *   - Capacity: Always equals rows * cols (dense, no gaps)
 *   - Iteration: for(i) for(j) accesses memory sequentially (cache-optimal)
 *
 * MEMORY OWNERSHIP:
 *   Caller owns both the struct and the values array.
 *   Must deallocate using dm_destroy() or manually: free(m->values); free(m);
 *
 * THREAD-SAFETY:
 *   - dm_get(): Safe if no concurrent writes to the matrix.
 *   - dm_set(): NOT safe if concurrent writes to *same* (i,j) cell.
 *                Race conditions possible. Serialize access or use locks.
 *   - Global state: dm_set_random_seed() modifies global RNG; NOT thread-safe.
 *
 * PRECISION:
 *   Uses IEEE 754 double precision (64-bit float).
 */
typedef struct DoubleMatrix {
  size_t rows;  /**< Number of rows. */
  size_t cols;  /**< Number of columns. */
  size_t capacity;  /**< Total allocated elements (= rows * cols for dense). */
  double *values;  /**< Row-major array of doubles. values[i*cols+j] = mat[i][j] */
} DoubleMatrix;

/**************************************/
/*         Matrix Creation            */
/**************************************/

/**
 * @brief Create an empty matrix with NULL values pointer.
 *
 * SEMANTICS:
 *   - Allocates only the struct, not the values array.
 *   - Useful for lazy initialization or placeholder.
 *
 * OWNERSHIP:
 *   Caller must free the struct with dm_destroy().
 *
 * @return Pointer to DoubleMatrix, or NULL if malloc fails.
 *
 * @see dm_destroy, dm_create
 */
DoubleMatrix *dm_create_empty(void);

/**
 * @brief Create matrix from existing values array (stores pointer, not copy).
 *
 * SEMANTICS:
 *   - Function stores a POINTER to values array.
 *   - Does NOT copy the array; caller retains ownership of the original array.
 *   - Modifying the original array WILL modify the matrix.
 *   - Freeing the original array while matrix exists = DANGLING POINTER.
 *
 * OWNERSHIP:
 *   Struct: Caller owns.
 *   Values array: Still owned by caller. Caller must NOT free until matrix freed.
 *
 * ALLOCATION ORDER:
 *   Implicit: values array owned by caller (passed pointer).
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param values Pointer to pre-allocated array of size rows*cols (assumed contiguous).
 *
 * @return Pointer to DoubleMatrix, or NULL if malloc fails.
 *
 * ⚠️ WARNING: Dangling pointer risk if caller frees values array early.
 *
 * @see dm_destroy, dm_create
 */
DoubleMatrix *dm_create_with_values(size_t rows, size_t cols, double *values);

/**
 * @brief Create uninitialized matrix of given dimensions.
 *
 * SEMANTICS:
 *   - Allocates struct + values array (rows*cols doubles).
 *   - Values are UNINITIALIZED (contain garbage; caller must initialize).
 *   - Capacity is fixed at rows*cols.
 *
 * OWNERSHIP:
 *   Caller owns both struct and values array.
 *   Must free using dm_destroy() or: free(m->values); free(m);
 *
 * ALLOCATION:
 *   Single malloc for struct+values (not separate allocations).
 *
 * ERROR HANDLING:
 *   Returns NULL if malloc fails (memory exhausted).
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 *
 * @return Pointer to DoubleMatrix, or NULL if allocation failed.
 *
 * @see dm_destroy, dm_create_empty, dm_create_with_values
 */
DoubleMatrix *dm_create(size_t rows, size_t cols);

/**
 * @brief Create deep copy of a matrix.
 *
 * SEMANTICS:
 *   - Allocates new struct + new values array.
 *   - Copies all values from original matrix.
 *   - Modifications to copy do NOT affect original.
 *
 * OWNERSHIP:
 *   Caller owns the cloned matrix and must free it.
 *
 * @param m Source matrix (const, not modified).
 *
 * @return New DoubleMatrix (independent), or NULL if allocation failed.
 *
 * @see dm_destroy
 */
DoubleMatrix *dm_clone(const DoubleMatrix *m);
MMATRIX_DEPRECATED("Use dm_clone instead")
DoubleMatrix *dm_create_clone(const DoubleMatrix *m);

/**
 * @brief Create identity matrix (1s on diagonal, 0s elsewhere).
 *
 * SEMANTICS:
 *   - Creates square n×n matrix.
 *   - Diagonal elements = 1.0; off-diagonal = 0.0.
 *
 * @param n Size of identity matrix (n×n).
 *
 * @return n×n identity matrix, or NULL if allocation failed.
 *
 * @see dm_destroy
 */
DoubleMatrix *dm_create_identity(size_t n);

/**
 * @brief Create matrix with random values (uniform [0, 1)).
 *
 * SEMANTICS:
 *   - Uses global random seed (set via dm_set_random_seed).
 *   - Non-deterministic by default (if seed not explicitly set).
 *   - Use dm_create_random_seeded() for reproducible results.
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (modifies global RNG state).
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 *
 * @return Matrix with random values, or NULL if allocation failed.
 *
 * @see dm_create_random_seeded, dm_set_random_seed
 */
DoubleMatrix *dm_create_random(size_t rows, size_t cols);

/**
 * @brief Create matrix with random values using explicit seed (deterministic).
 *
 * SEMANTICS:
 *   - Seed directly controls all generated values.
 *   - Same seed → Same matrix output (reproducible).
 *   - Uses SplitMix64 mixing for deterministic, uniform random generation.
 *   - Does NOT modify global random seed; local to this call.
 *
 * THREAD-SAFETY:
 *   Thread-safe (no global state modified).
 *   Multiple threads can call concurrently with different seeds.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param seed Seed value (uint64_t). Same seed → same output.
 *
 * @return Matrix with deterministic random values, or NULL if allocation failed.
 *
 * @see dm_set_random_seed, dm_create_random
 */
DoubleMatrix *dm_create_random_seeded(size_t rows, size_t cols, uint64_t seed);

/**
 * @brief Set global random seed for dm_create_random().
 *
 * SEMANTICS:
 *   - Subsequent dm_create_random() calls use this seed + internal counter.
 *   - NOT used by dm_create_random_seeded() (which is truly independent).
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (modifies global state).
 *   Race condition: concurrent calls may cause unexpected results.
 *   Recommendation: Serialize seed setting or use _seeded() variant.
 *
 * @param seed Seed value for global RNG.
 *
 * @see dm_get_random_seed, dm_create_random, dm_create_random_seeded
 */
void dm_set_random_seed(uint64_t seed);

/**
 * @brief Get current global random seed.
 *
 * @return Current global seed value.
 *
 * @see dm_set_random_seed
 */
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

/**
 * @brief Get element at (i, j).
 *
 * SEMANTICS:
 *   - Safe read access.
 *   - Uses row-major index: values[i * cols + j].
 *   - No bounds checking (undefined behavior if i >= rows or j >= cols).
 *
 * @param mat Matrix (const, not modified).
 * @param i Row index (0-based).
 * @param j Column index (0-based).
 *
 * @return Element value at (i, j).
 *
 * ⚠️ WARNING: No bounds checking. Out-of-bounds access = undefined behavior.
 *
 * @see dm_set
 */
double dm_get(const DoubleMatrix *mat, size_t i, size_t j);

/**
 * @brief Set element at (i, j).
 *
 * SEMANTICS:
 *   - Modifies matrix in-place.
 *   - No bounds checking.
 *
 * THREAD-SAFETY:
 *   NOT safe if concurrent accesses to same (i,j) cell occur.
 *   Can race: multiple threads writing simultaneously = unpredictable result.
 *
 * @param mat Matrix (modified).
 * @param i Row index (0-based).
 * @param j Column index (0-based).
 * @param value New element value.
 *
 * @see dm_get
 */
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value);
DoubleMatrix *dm_get_row(const DoubleMatrix *mat, size_t i);
DoubleMatrix *dm_get_last_row(const DoubleMatrix *mat);
DoubleMatrix *dm_get_col(const DoubleMatrix *mat, size_t j);
DoubleMatrix *dm_get_last_col(const DoubleMatrix *mat);

/**************************************/
/*         Matrix Shape Ops           */
/**************************************/
void dm_reshape(DoubleMatrix *matrix, size_t new_rows, size_t new_cols);
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
void dm_inplace_add(DoubleMatrix *mat1, const DoubleMatrix *mat2);
void dm_inplace_diff(DoubleMatrix *mat1, const DoubleMatrix *mat2);
void dm_inplace_transpose(DoubleMatrix *mat);
void dm_inplace_multiply_by_number(DoubleMatrix *mat, const double scalar);
void dm_inplace_gauss_elimination(DoubleMatrix *mat);
void dm_inplace_elementwise_multiply(DoubleMatrix *mat1,
                                     const DoubleMatrix *mat2);
void dm_inplace_div(DoubleMatrix *mat1, const DoubleMatrix *mat2);

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
double *dm_to_column_major(const DoubleMatrix *mat);

/**************************************/
/*              File I/O              */
/**************************************/
void dm_print(const DoubleMatrix *matrix);
const char *dm_active_library(void);

/**************************************/
/*         Memory Management          */
/**************************************/

/**
 * @brief Free a matrix and all associated memory.
 *
 * SEMANTICS:
 *   - Deallocates both struct and values array.
 *   - When called on matrix created with dm_create_with_values(),
 *     frees the struct but NOT the values array (caller owns that).
 *   - Safe to call on NULL (no-op).
 *
 * OWNERSHIP AFTER CALL:
 *   - Matrix pointer becomes INVALID (dangling pointer).
 *   - Caller must NOT access after dm_destroy().
 *
 * DEALLOCATION ORDER (internal):
 *   1. free(mat->values) [only if not from dm_create_with_values]
 *   2. free(mat)
 *
 * ⚠️ CAUTION: Do NOT call twice on same pointer (double-free crash).
 *
 * @param mat Matrix to deallocate (modified to invalid state).
 *
 * @see dm_create, dm_create_empty
 */
void dm_destroy(DoubleMatrix *mat);

#endif  // DM_H
