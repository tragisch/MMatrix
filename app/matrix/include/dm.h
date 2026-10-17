/**
 * @file dm.h
 * @brief Public API for dense double-precision matrices.
 */

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

/** @brief Dense double matrix with row-major storage: `values[i * cols + j]`. */
typedef struct DoubleMatrix {
  size_t rows;
  size_t cols;
  size_t capacity;
  double *values;
} DoubleMatrix;

/**************************************/
/*         Matrix Creation            */
/**************************************/

/**
 * @brief Create empty matrix metadata (`values == NULL`).
 * @return Empty matrix metadata (caller must set `values`), or NULL on allocation failure.
 */
DoubleMatrix *dm_create_empty(void);

/**
 * @brief Create matrix from external values pointer (no copy).
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param values Pointer to existing row-major storage.
 * @return New matrix wrapping @p values, or NULL on allocation failure.
 */
DoubleMatrix *dm_create_with_values(size_t rows, size_t cols, double *values);

/**
 * @brief Create uninitialized matrix with shape `rows x cols`.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return New uninitialized matrix, or NULL on allocation failure.
 */
DoubleMatrix *dm_create(size_t rows, size_t cols);

/**
 * @brief Create deep copy of a matrix.
 * @param m Source matrix.
 * @return New cloned matrix, or NULL on allocation failure.
 */
DoubleMatrix *dm_clone(const DoubleMatrix *m);
MMATRIX_DEPRECATED("Use dm_clone instead")
DoubleMatrix *dm_create_clone(const DoubleMatrix *m);

/**
 * @brief Create identity matrix with shape `n x n`.
 * @param n Dimension.
 * @return New identity matrix, or NULL on allocation failure.
 */
DoubleMatrix *dm_create_identity(size_t n);

/**
 * @brief Create random matrix using global RNG state (not thread-safe).
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return New random matrix, or NULL on allocation failure.
 */
DoubleMatrix *dm_create_random(size_t rows, size_t cols);

/**
 * @brief Create deterministic random matrix from explicit seed.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param seed Random seed.
 * @return New random matrix, or NULL on allocation failure.
 */
DoubleMatrix *dm_create_random_seeded(size_t rows, size_t cols, uint64_t seed);

/**
 * @brief Set global RNG seed used by non-seeded random creators (not thread-safe).
 * @param seed Random seed.
 */
void dm_set_random_seed(uint64_t seed);

/**
 * @brief Get current global RNG seed.
 * @return Current global RNG seed.
 */
uint64_t dm_get_random_seed(void);

/**************************************/
/*         Matrix Import              */
/**************************************/
/**
 * @brief Create matrix by copying data from row-pointer array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param array Array of row pointers.
 * @return New matrix, or NULL on allocation failure.
 */
DoubleMatrix *dm_from_array_ptrs(size_t rows, size_t cols, double **array);
/**
 * @brief Create matrix by copying data from static 2D C array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param array VLA static 2D array.
 * @return New matrix, or NULL on allocation failure.
 */
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
 * @brief Read element at `(i, j)`; caller must ensure valid bounds.
 * @param mat Source matrix.
 * @param i Row index.
 * @param j Column index.
 * @return Element value.
 */
double dm_get(const DoubleMatrix *mat, size_t i, size_t j);

/**
 * @brief Write element at `(i, j)`; concurrent writes to same matrix are not thread-safe.
 * @param mat Destination matrix.
 * @param i Row index.
 * @param j Column index.
 * @param value Element value to write.
 */
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value);
/**
 * @brief Return row `i` as new matrix.
 * @param mat Source matrix.
 * @param i Row index.
 * @return New 1×cols matrix, or NULL on error.
 */
DoubleMatrix *dm_get_row(const DoubleMatrix *mat, size_t i);
/**
 * @brief Return last row as new matrix.
 * @param mat Source matrix.
 * @return New 1×cols matrix, or NULL on error.
 */
DoubleMatrix *dm_get_last_row(const DoubleMatrix *mat);
/**
 * @brief Return column `j` as new matrix.
 * @param mat Source matrix.
 * @param j Column index.
 * @return New rows×1 matrix, or NULL on error.
 */
DoubleMatrix *dm_get_col(const DoubleMatrix *mat, size_t j);
/**
 * @brief Return last column as new matrix.
 * @param mat Source matrix.
 * @return New rows×1 matrix, or NULL on error.
 */
DoubleMatrix *dm_get_last_col(const DoubleMatrix *mat);

/**************************************/
/*         Matrix Shape Ops           */
/**************************************/
/**
 * @brief Reshape matrix metadata; element count must remain compatible.
 * @param matrix Matrix to reshape.
 * @param new_rows New row count.
 * @param new_cols New column count.
 */
void dm_reshape(DoubleMatrix *matrix, size_t new_rows, size_t new_cols);
/**
 * @brief Resize matrix storage to new shape.
 * @param mat Matrix to resize.
 * @param new_row New row count.
 * @param new_col New column count.
 */
void dm_resize(DoubleMatrix *mat, size_t new_row, size_t new_col);

/**************************************/
/*       Matrix Arithmetic            */
/**************************************/
/**
 * @brief Multiply two dense double matrices.
 * @param mat1 Left matrix.
 * @param mat2 Right matrix.
 * @return New product matrix, or NULL on error.
 */
DoubleMatrix *dm_multiply(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
/**
 * @brief Multiply matrix by scalar value.
 * @param mat Input matrix.
 * @param number Scalar multiplier.
 * @return New scaled matrix, or NULL on error.
 */
DoubleMatrix *dm_multiply_by_number(const DoubleMatrix *mat,
                                    const double number);
/**
 * @brief Element-wise product of two matrices.
 * @param mat1 First matrix.
 * @param mat2 Second matrix.
 * @return New element-wise product, or NULL on error.
 */
DoubleMatrix *dm_elementwise_multiply(const DoubleMatrix *mat1,
                                      const DoubleMatrix *mat2);
/**
 * @brief Element-wise division of two matrices.
 * @param mat1 Dividend matrix.
 * @param mat2 Divisor matrix.
 * @return New element-wise quotient, or NULL on error.
 */
DoubleMatrix *dm_div(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
/**
 * @brief Add two matrices.
 * @param mat1 First matrix.
 * @param mat2 Second matrix.
 * @return New sum matrix, or NULL on error.
 */
DoubleMatrix *dm_add(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
/**
 * @brief Subtract two matrices (`mat1 - mat2`).
 * @param mat1 Minuend matrix.
 * @param mat2 Subtrahend matrix.
 * @return New difference matrix, or NULL on error.
 */
DoubleMatrix *dm_diff(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
/**
 * @brief Compute inverse matrix.
 * @param mat Input matrix.
 * @return New inverse matrix, or NULL on error.
 */
DoubleMatrix *dm_inverse(const DoubleMatrix *mat);

/**************************************/
/*       Matrix Transformations       */
/**************************************/
/**
 * @brief Return transposed copy of a matrix.
 * @param mat Input matrix.
 * @return New transposed matrix, or NULL on error.
 */
DoubleMatrix *dm_transpose(const DoubleMatrix *mat);

/**************************************/
/*        In-place Operations         */
/**************************************/
/**
 * @brief In-place matrix addition.
 * @param mat1 First matrix (modified).
 * @param mat2 Second matrix.
 * @retval true Success.
 * @retval false Invalid input or shape mismatch.
 */
bool dm_inplace_add(DoubleMatrix *mat1, const DoubleMatrix *mat2);
/**
 * @brief In-place matrix subtraction (`mat1 -= mat2`).
 * @param mat1 Minuend matrix (modified).
 * @param mat2 Subtrahend matrix.
 * @retval true Success.
 * @retval false Invalid input or shape mismatch.
 */
bool dm_inplace_diff(DoubleMatrix *mat1, const DoubleMatrix *mat2);
/**
 * @brief In-place transpose (implementation-dependent constraints).
 * @param mat Matrix to transpose (modified in-place).
 * @retval true Success.
 * @retval false Invalid input.
 */
bool dm_inplace_transpose(DoubleMatrix *mat);
/**
 * @brief In-place scalar multiplication.
 * @param mat Matrix to scale (modified).
 * @param scalar Multiplier value.
 * @retval true Success.
 * @retval false Invalid input.
 */
bool dm_inplace_multiply_by_number(DoubleMatrix *mat, const double scalar);
/**
 * @brief In-place Gaussian elimination transform.
 * @param mat Matrix to transform (modified).
 * @retval true Success.
 * @retval false Matrix is singular or invalid.
 */
bool dm_inplace_gauss_elimination(DoubleMatrix *mat);
/**
 * @brief In-place element-wise multiplication.
 * @param mat1 First matrix (modified).
 * @param mat2 Second matrix.
 * @retval true Success.
 * @retval false Invalid input or shape mismatch.
 */
bool dm_inplace_elementwise_multiply(DoubleMatrix *mat1,
                                     const DoubleMatrix *mat2);
/**
 * @brief In-place element-wise division.
 * @param mat1 Dividend matrix (modified).
 * @param mat2 Divisor matrix.
 * @retval true Success.
 * @retval false Invalid input or shape mismatch.
 */
bool dm_inplace_div(DoubleMatrix *mat1, const DoubleMatrix *mat2);

/**************************************/
/*         Matrix Properties          */
/**************************************/
/**
 * @brief Determinant of a square matrix.
 * @param mat Input matrix.
 * @return Determinant value.
 */
double dm_determinant(const DoubleMatrix *mat);
/**
 * @brief Trace of a square matrix.
 * @param mat Input matrix.
 * @return Trace value (sum of diagonal elements).
 */
double dm_trace(const DoubleMatrix *mat);
/**
 * @brief Matrix rank.
 * @param mat Input matrix.
 * @return Numerical rank.
 */
size_t dm_rank(const DoubleMatrix *mat);
/**
 * @brief Matrix norm (implementation-defined norm type).
 * @param mat Input matrix.
 * @return Norm value.
 */
double dm_norm(const DoubleMatrix *mat);
/**
 * @brief Matrix density in range `[0,1]`.
 * @param mat Input matrix.
 * @return Density value (approx. ratio of non-zero elements).
 */
double dm_density(const DoubleMatrix *mat);

/**************************************/
/*     Matrix Property Checks         */
/**************************************/
/**
 * @brief Check whether matrix is empty/uninitialized.
 * @param mat Input matrix.
 * @retval true Matrix is empty.
 * @retval false Matrix is valid.
 */
bool dm_is_empty(const DoubleMatrix *mat);
/**
 * @brief Check whether matrix is square.
 * @param mat Input matrix.
 * @retval true Matrix is square.
 * @retval false Otherwise.
 */
bool dm_is_square(const DoubleMatrix *mat);
/**
 * @brief Check whether matrix represents a vector (one dimension equals 1).
 * @param mat Input matrix.
 * @retval true Matrix is a vector.
 * @retval false Otherwise.
 */
bool dm_is_vector(const DoubleMatrix *mat);
/**
 * @brief Check whether two matrices have equal shape.
 * @param mat1 First matrix.
 * @param mat2 Second matrix.
 * @retval true Shapes match.
 * @retval false Shapes differ.
 */
bool dm_is_equal_size(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
/**
 * @brief Check whether two matrices are element-wise equal.
 * @param mat1 First matrix.
 * @param mat2 Second matrix.
 * @retval true All elements match.
 * @retval false Elements differ or shapes mismatch.
 */
bool dm_is_equal(const DoubleMatrix *mat1, const DoubleMatrix *mat2);

/**************************************/
/*         Matrix Utilities           */
/**************************************/
/**
 * @brief Export matrix as newly allocated column-major array (caller must free).
 * @param mat Input matrix.
 * @return Newly allocated array, or NULL on error.
 */
double *dm_to_column_major(const DoubleMatrix *mat);

/**************************************/
/*              File I/O              */
/**************************************/
/**
 * @brief Print matrix to stdout (debug helper).
 * @param matrix Matrix to print.
 */
void dm_print(const DoubleMatrix *matrix);
/**
 * @brief Return active compute backend name.
 * @return Backend name string (statically allocated).
 */
const char *dm_active_library(void);

/**************************************/
/*         Memory Management          */
/**************************************/

/**
 * @brief Destroy matrix (NULL-safe).
 * @param mat Matrix to destroy.
 */
void dm_destroy(DoubleMatrix *mat);

#endif  // DM_H
