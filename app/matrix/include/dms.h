/**
 * @file dms.h
 * @brief Public API for sparse double matrices in COO format.
 */

/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef DMS_SPARSE_H
#define DMS_SPARSE_H

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

/** @brief Forward declaration for SuiteSparse type when `cs.h` is unavailable. */
#ifndef DMS_HAS_CSPARSE
typedef struct cs_di_sparse cs;
#endif

/**
 * @brief Sparse COO matrix stored as `(row, col, value)` triples.
 * @details COO is the builder format; CSC/CSR caches are built lazily and
 * invalidated on COO mutations.
 */
typedef struct DoubleSparseMatrix {
  size_t rows;
  size_t cols;
  size_t nnz;
  size_t capacity;
  size_t *row_indices;
  size_t *col_indices;
  double *values;
  cs *csc_cache;
  bool csc_valid;
  size_t *csr_cache;   // lazily computed row_offsets[rows+1], NULL if invalid
  bool csr_valid;
} DoubleSparseMatrix;

/**
 * @brief Create empty sparse matrix metadata (`nnz = 0`, arrays `NULL`).
 * @return Empty sparse matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_create_empty(void);

/**
 * @brief Create sparse matrix by copying external COO arrays.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param nnz Number of non-zero entries.
 * @param row_indices Array of row indices (length nnz).
 * @param col_indices Array of column indices (length nnz).
 * @param values Array of values (length nnz).
 * @return Sparse matrix with copied COO data, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_create_with_values(size_t rows, size_t cols, size_t nnz,
                                           size_t *row_indices,
                                           size_t *col_indices, double *values);

/**
 * @brief Create empty COO matrix with pre-allocated triple capacity.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param capacity Pre-allocated capacity for nnz triples.
 * @return Empty sparse matrix with allocated arrays, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_create(size_t rows, size_t cols, size_t capacity);

/**
 * @brief Create deep copy of sparse matrix.
 * @param m Source sparse matrix.
 * @return New cloned sparse matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_clone(const DoubleSparseMatrix *m);
/**
 * @deprecated Use dms_clone instead.
 * @param m Source sparse matrix.
 * @return New cloned sparse matrix, or NULL on allocation failure.
 */
MMATRIX_DEPRECATED("Use dms_clone instead")
DoubleSparseMatrix *dms_create_clone(const DoubleSparseMatrix *m);

/**
 * @brief Create sparse identity matrix with shape `n x n`.
 * @param n Dimension (rows and columns).
 * @return Identity sparse matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_create_identity(size_t n);

/**
 * @brief Create sparse random matrix using shared global RNG state.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param density Target sparsity as fraction in [0,1].
 * @note Concurrent callers must synchronize externally.
 * @return Random sparse matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_create_random(size_t rows, size_t cols, double density);

/**
 * @brief Create deterministic sparse random matrix from explicit seed.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param density Target sparsity as fraction in [0,1].
 * @param seed Random seed for reproducibility.
 * @return Deterministic random sparse matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_create_random_seeded(size_t rows, size_t cols,
                                              double density, uint64_t seed);

/**
 * @brief Set shared global RNG seed used by non-seeded random creators.
 * @param seed Random seed value.
 * @note Concurrent callers must synchronize externally.
 */
void dms_set_random_seed(uint64_t seed);

/**
 * @brief Get current global RNG seed.
 * @return Current global RNG seed value.
 */
uint64_t dms_get_random_seed(void);

/**
 * @brief Convert COO sparse matrix to SuiteSparse `cs` matrix.
 * @param coo Source COO matrix.
 * @return New SuiteSparse `cs` matrix, or NULL on allocation failure.
 */
cs *dms_to_cs(const DoubleSparseMatrix *coo);
/**
 * @brief Convert SuiteSparse `cs` matrix to COO sparse matrix.
 * @param A Source SuiteSparse `cs` matrix.
 * @return New COO sparse matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_from_cs(const cs *A);
/**
 * @deprecated Use dms_from_cs instead.
 * @param A Source SuiteSparse `cs` matrix.
 * @return New COO sparse matrix, or NULL on allocation failure.
 */
MMATRIX_DEPRECATED("Use dms_from_cs instead")
DoubleSparseMatrix *cs_to_dms(const cs *A);
/**
 * @brief Export sparse matrix as newly allocated dense row-major array.
 * @param mat Source sparse matrix.
 * @return Newly allocated dense array (size rows x cols), or NULL on allocation failure. Caller must free.
 */
double *dms_to_array(const DoubleSparseMatrix *mat);

/**
 * @brief Create sparse matrix from dense row-major array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param array Dense row-major array of size rows x cols.
 * @return Sparse matrix with zero entries trimmed, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_create_from_array(size_t rows, size_t cols,
                                          double *array);
/**
 * @brief Create sparse matrix from static dense 2D C array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param array VLA static 2D array of shape rows x cols.
 * @return Sparse matrix with zero entries trimmed, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_from_array_static(size_t rows, size_t cols,
                                          double array[rows][cols]);
/**
 * @deprecated Use dms_from_array_static instead.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param array VLA static 2D array.
 * @return Sparse matrix, or NULL on allocation failure.
 */
MMATRIX_DEPRECATED("Use dms_from_array_static instead")
DoubleSparseMatrix *dms_create_from_2D_array(size_t rows, size_t cols,
                                             double array[rows][cols]);

/**
 * @brief Insert or update COO triple `(i, j, value)`.
 * @param mat Sparse matrix to update.
 * @param i Row index (0-based, must be < rows).
 * @param j Column index (0-based, must be < cols).
 * @param value New value (replaces or adds to existing).
 * @details Internal ordering may be deferred, but sorted COO order is restored before binary-search access or CSparse conversion.
 * @return true on success, false on allocation/bound error.
 */
bool dms_set(DoubleSparseMatrix *mat, size_t i, size_t j, double value);

/**
 * @brief Read value at first matching `(i, j)` position.
 * @param mat Source sparse matrix.
 * @param i Row index (0-based, must be < rows).
 * @param j Column index (0-based, must be < cols).
 * @return Value at (i, j), or 0.0 if no entry exists.
 */
double dms_get(const DoubleSparseMatrix *mat, size_t i, size_t j);

/**
 * @brief Return row `i` as new sparse matrix.
 * @param mat Source sparse matrix.
 * @param i Row index (0-based, must be < rows).
 * @return New sparse 1 x cols matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_get_row(const DoubleSparseMatrix *mat, size_t i);
/**
 * @brief Return last row as new sparse matrix.
 * @param mat Source sparse matrix.
 * @return New sparse 1 x cols matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_get_last_row(const DoubleSparseMatrix *mat);
/**
 * @brief Return column `j` as new sparse matrix.
 * @param mat Source sparse matrix.
 * @param j Column index (0-based, must be < cols).
 * @return New sparse rows x 1 matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_get_col(const DoubleSparseMatrix *mat, size_t j);
/**
 * @brief Return last column as new sparse matrix.
 * @param mat Source sparse matrix.
 * @return New sparse rows x 1 matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_get_last_col(const DoubleSparseMatrix *mat);

/**
 * @brief Multiply two sparse matrices.
 * @param mat1 Left sparse matrix (shape: m x k).
 * @param mat2 Right sparse matrix (shape: k x n).
 * @return Result sparse matrix of shape m x n, or NULL on allocation/shape mismatch.
 */
DoubleSparseMatrix *dms_multiply(const DoubleSparseMatrix *mat1,
                                 const DoubleSparseMatrix *mat2);
/**
 * @brief Multiply sparse matrix by scalar value.
 * @param mat Source sparse matrix.
 * @param number Scalar multiplier.
 * @return Result sparse matrix, or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_multiply_by_number(const DoubleSparseMatrix *mat,
                                           const double number);
/**
 * @brief Return transposed sparse matrix.
 * @param mat Source sparse matrix (shape: rows x cols).
 * @return Transposed sparse matrix (shape: cols x rows), or NULL on allocation failure.
 */
DoubleSparseMatrix *dms_transpose(const DoubleSparseMatrix *mat);

/**
 * @brief Sparse matrix-vector product `y = A * x`.
 * @param mat Source sparse matrix (shape: rows x cols).
 * @param x Input vector (length cols).
 * @param y Output vector (length rows); pre-allocated by caller.
 * @pre `x` has `mat->cols` elements and `y` has `mat->rows` elements.
 * @return true on success, false on dimension/allocation error.
 */
bool dms_spmv(const DoubleSparseMatrix *mat, const double *x, double *y);

/**
 * @brief Matrix density in range `[0,1]`.
 * @param mat Source sparse matrix.
 * @return Ratio of non-zero entries to total entries (nnz / (rows * cols)).
 */
double dms_density(const DoubleSparseMatrix *mat);

/**
 * @brief Print sparse matrix to stdout (debug helper).
 * @param mat Matrix to print.
 */
void dms_print(const DoubleSparseMatrix *mat);

/**
 * @brief Reallocate COO arrays to `new_capacity` (`new_capacity >= nnz`).
 * @param mat Sparse matrix to reallocate.
 * @param new_capacity New capacity for triples.
 * @return true on success, false on allocation failure.
 */
bool dms_realloc(DoubleSparseMatrix *mat, size_t new_capacity);

/**
 * @brief Destroy sparse matrix (NULL-safe).
 * @param mat Sparse matrix pointer (NULL-safe; no-op if NULL).
 */
void dms_destroy(DoubleSparseMatrix *mat);

#endif  // DMS_SPARSE_H
