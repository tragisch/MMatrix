/**
 * @file sm.h
 * @brief Public API for dense single-precision matrices and backend-dispatched kernels.
 */

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
/*       Compute Backend Selection    */
/**************************************/

/** @brief Available compute backends (runtime selection). */
typedef enum SmBackend {
  SM_BACKEND_DEFAULT = 0,     // best available: Accelerate > OpenBLAS > OpenMP
  SM_BACKEND_ACCELERATE = 1,  // Apple Accelerate (AMX)
  SM_BACKEND_MPS = 2,         // Metal Performance Shaders (GPU)
  SM_BACKEND_OPENBLAS = 3,    // OpenBLAS
  SM_BACKEND_OPENMP = 4,      // OpenMP / ARM NEON (no BLAS)
} SmBackend;

/**
 * @brief Set active compute backend for dispatched operations.
 * @retval true Backend selected.
 * @retval false Backend not available in current build.
 */
bool sm_set_backend(SmBackend backend);

/**
 * @brief Get currently active compute backend.
 * @return Active backend enum value.
 */
SmBackend sm_get_backend(void);

/**************************************/
/*         Float Matrix Type          */
/**************************************/

/** @brief Dense float matrix with row-major storage: `values[i * cols + j]`. */
typedef struct FloatMatrix {
  size_t rows;
  size_t cols;
  size_t capacity;
  float *values;
} FloatMatrix;

/**************************************/
/*         Matrix Creation            */
/**************************************/

/**
 * @brief Create empty matrix metadata (`values == NULL`).
 * @return Empty matrix with uninitialized rows/cols, or NULL on allocation failure.
 */
FloatMatrix *sm_create_empty(void);

/**
 * @brief Create zero-initialized matrix with shape `rows x cols`.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return New zero matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_create_zeros(size_t rows, size_t cols);

/**
 * @brief Create uninitialized matrix with shape `rows x cols`.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return New uninitialized matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_create(size_t rows, size_t cols);

/**
 * @brief Create matrix by copying provided values array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param values Pointer to row-major float array of size `rows * cols`.
 * @return New matrix with copied values, or NULL on allocation failure.
 */
FloatMatrix *sm_create_with_values(size_t rows, size_t cols, float *values);

/**
 * @brief Create deep copy of a matrix.
 * @param m Source matrix pointer.
 * @return New cloned matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_clone(const FloatMatrix *m);

/**
 * @brief Create identity matrix with shape `n x n`.
 * @param n Dimension (rows and columns).
 * @return Identity matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_create_identity(size_t n);

/**
 * @brief Create random matrix using global RNG state (not thread-safe).
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return Random matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_create_random(size_t rows, size_t cols);

/**
 * @brief Create deterministic random matrix from explicit seed.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param seed Random seed for reproducibility.
 * @return Random matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_create_random_seeded(size_t rows, size_t cols, uint64_t seed);

/**
 * @brief Set global RNG seed used by non-seeded creators (not thread-safe).
 * @param seed Random seed value.
 */
void sm_set_random_seed(uint64_t seed);

/**
 * @brief Get current global RNG seed.
 * @return Current global RNG seed value.
 */
uint64_t sm_get_random_seed(void);

/**
 * @brief Create He-initialized weight matrix (ReLU-like networks).
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param fan_in Fan-in (typically input dimension).
 * @return He-initialized weight matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_create_random_he(size_t rows, size_t cols, size_t fan_in);

/**
 * @brief Create Xavier/Glorot-initialized weight matrix.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param fan_in Fan-in (input dimension).
 * @param fan_out Fan-out (output dimension).
 * @return Xavier-initialized weight matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_create_random_xavier(size_t rows, size_t cols, size_t fan_in,
                                     size_t fan_out);

/**
 * @brief Create matrix by copying data from row-pointer array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param array Array of row pointers.
 * @return New matrix with copied data, or NULL on allocation failure.
 */
FloatMatrix *sm_from_array_ptrs(size_t rows, size_t cols, float **array);

/**
 * @brief Create matrix by copying data from static 2D C array.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param array VLA static 2D array of shape rows x cols.
 * @return New matrix with copied data, or NULL on allocation failure.
 */
FloatMatrix *sm_from_array_static(size_t rows, size_t cols,
                                  float array[rows][cols]);

/**
 * @brief Export matrix to newly allocated row-major array (caller must free).
 * @param matrix Source matrix.
 * @return Newly allocated float array in row-major order, or NULL on allocation failure.
 */
float *sm_to_array(FloatMatrix *matrix);
/**
 * @deprecated Use sm_to_array instead.
 * @param matrix Source matrix.
 * @return Newly allocated float array, or NULL on allocation failure.
 */
MMATRIX_DEPRECATED("Use sm_to_array instead")
float *sm_create_array_from_matrix(FloatMatrix *matrix);

/**************************************/
/*         Matrix Accessors           */
/**************************************/

/**
 * @brief Read element at `(i, j)`; caller must ensure valid bounds.
 * @param mat Source matrix.
 * @param i Row index (0-based, must be < rows).
 * @param j Column index (0-based, must be < cols).
 * @return Element value at position (i, j).
 */
float sm_get(const FloatMatrix *mat, size_t i, size_t j);

/**
 * @brief Write element at `(i, j)`; concurrent writes are not thread-safe.
 * @param mat Destination matrix.
 * @param i Row index (0-based, must be < rows).
 * @param j Column index (0-based, must be < cols).
 * @param value New element value.
 */
void sm_set(FloatMatrix *mat, size_t i, size_t j, float value);
/**
 * @brief Return row `i` as new matrix.
 * @param mat Source matrix.
 * @param i Row index (0-based, must be < rows).
 * @return New 1 x cols matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_get_row(const FloatMatrix *mat, size_t i);
/**
 * @brief Return last row as new matrix.
 * @param mat Source matrix.
 * @return New 1 x cols matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_get_last_row(const FloatMatrix *mat);
/**
 * @brief Return column `j` as new matrix.
 * @param mat Source matrix.
 * @param j Column index (0-based, must be < cols).
 * @return New rows x 1 matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_get_col(const FloatMatrix *mat, size_t j);
/**
 * @brief Return last column as new matrix.
 * @param mat Source matrix.
 * @return New rows x 1 matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_get_last_col(const FloatMatrix *mat);
/**
 * @brief Return row slice `[start, end)` as new matrix.
 * @param mat Source matrix.
 * @param start Starting row index (inclusive, 0-based).
 * @param end Ending row index (exclusive, 0-based).
 * @return New (end-start) x cols matrix, or NULL on allocation/range error.
 */
FloatMatrix *sm_slice_rows(const FloatMatrix *mat, size_t start, size_t end);

/**************************************/
/*         Matrix Shape Ops           */
/**************************************/
/**
 * @brief Reshape matrix metadata; element count must remain compatible.
 * @param matrix Matrix to reshape (metadata only, not realloc).
 * @param new_rows New number of rows.
 * @param new_cols New number of columns.
 */
void sm_reshape(FloatMatrix *matrix, size_t new_rows, size_t new_cols);
/**
 * @brief Resize matrix storage to new shape.
 * @param mat Matrix to resize (may reallocate).
 * @param new_row New number of rows.
 * @param new_col New number of columns.
 */
void sm_resize(FloatMatrix *mat, size_t new_row, size_t new_col);

/**************************************/
/*       Matrix Transformations       */
/**************************************/
/**
 * @brief Return transposed copy of matrix.
 * @param mat Source matrix (shape: rows x cols).
 * @return Transposed matrix (shape: cols x rows), or NULL on allocation failure.
 */
FloatMatrix *sm_transpose(const FloatMatrix *mat);

/**************************************/
/*       Matrix Arithmetic            */
/**************************************/

/**
 * @brief Add two matrices element-wise.
 * @param mat1 First matrix.
 * @param mat2 Second matrix (must have same shape as mat1).
 * @return Result matrix, or NULL on allocation/shape mismatch.
 */
FloatMatrix *sm_add(const FloatMatrix *mat1, const FloatMatrix *mat2);
/**
 * @brief Subtract two matrices element-wise (`mat1 - mat2`).
 * @param mat1 First matrix.
 * @param mat2 Second matrix (must have same shape as mat1).
 * @return Result matrix, or NULL on allocation/shape mismatch.
 */
FloatMatrix *sm_diff(const FloatMatrix *mat1, const FloatMatrix *mat2);
/**
 * @brief Multiply two matrices (standard matrix multiplication).
 * @param mat1 Left matrix (shape: m x k).
 * @param mat2 Right matrix (shape: k x n).
 * @return Result matrix of shape m x n, or NULL on allocation/shape mismatch.
 */
FloatMatrix *sm_multiply(const FloatMatrix *mat1, const FloatMatrix *mat2);
/**
 * @brief Element-wise product of two matrices.
 * @param mat1 First matrix.
 * @param mat2 Second matrix (must have same shape as mat1).
 * @return Result matrix, or NULL on allocation/shape mismatch.
 */
FloatMatrix *sm_elementwise_multiply(const FloatMatrix *mat1,
                                     const FloatMatrix *mat2);
/**
 * @brief Multiply matrix by scalar value.
 * @param mat Source matrix.
 * @param number Scalar multiplier.
 * @return Result matrix, or NULL on allocation failure.
 */
FloatMatrix *sm_multiply_by_number(const FloatMatrix *mat, const float number);
/**
 * @brief Compute inverse matrix.
 * @param mat Source matrix (must be square and invertible).
 * @return Inverse matrix, or NULL on allocation/singularity error.
 */
FloatMatrix *sm_inverse(const FloatMatrix *mat);
/**
 * @brief Element-wise division of two matrices.
 * @param mat1 Numerator matrix.
 * @param mat2 Denominator matrix (must have same shape as mat1).
 * @return Result matrix, or NULL on allocation/shape/zero-division error.
 */
FloatMatrix *sm_div(const FloatMatrix *mat1, const FloatMatrix *mat2);
/**
 * @brief Solve linear system `A * x = b`.
 * @param A Coefficient matrix (must be square and invertible).
 * @param b Right-hand-side vector (shape: rows x 1).
 * @return Solution vector x, or NULL on allocation/singularity error.
 */
FloatMatrix *sm_solve_system(const FloatMatrix *A, const FloatMatrix *b);

/**************************************/
/*        Advanced Operations         */
/**************************************/
/** @brief Transpose mode for BLAS-style operations. */
typedef enum SmTranspose {
  SM_NO_TRANSPOSE = 0,
  SM_TRANSPOSE = 1,
} SmTranspose;

/**
 * @brief BLAS-style GEMM kernel.
 * @param C Output matrix (shape must be compatible with A and B).
 * @param alpha Scalar multiplier for A*B term.
 * @param A Left input matrix.
 * @param trans_a Transpose mode for A (SM_NO_TRANSPOSE or SM_TRANSPOSE).
 * @param B Right input matrix.
 * @param trans_b Transpose mode for B.
 * @param beta Scalar multiplier for existing C values.
 * @details Computes in-place: `C = alpha * op(A) * op(B) + beta * C`.
 * @return true on success, false on shape mismatch or allocation failure.
 */
bool sm_gemm(FloatMatrix *C, float alpha, const FloatMatrix *A,
             SmTranspose trans_a, const FloatMatrix *B, SmTranspose trans_b,
             float beta);

/**
 * @brief Fused kernel: GEMM + optional bias + ReLU activation.
 * @param C Output matrix to accumulate result into.
 * @param A Left input matrix.
 * @param trans_a Transpose mode for A.
 * @param B Right input matrix.
 * @param trans_b Transpose mode for B.
 * @param bias Bias vector (shape `1 x cols` or `rows x cols`, or NULL to skip).
 * @details Computes: `C += A * op(B); C += bias (broadcast); C = max(C, 0)`.
 * @return true on success, false on shape/allocation error.
 */
bool sm_gemm_bias_relu(FloatMatrix *C, const FloatMatrix *A, SmTranspose trans_a,
                       const FloatMatrix *B, SmTranspose trans_b,
                       const FloatMatrix *bias);

/**************************************/
/*        Matrix In-Place Ops         */
/**************************************/
/**
 * @brief In-place matrix addition.
 * @param mat1 Matrix to accumulate into.
 * @param mat2 Matrix to add (must have same shape as mat1).
 * @return true on success, false on shape/allocation error.
 */
bool sm_inplace_add(FloatMatrix *mat1, const FloatMatrix *mat2);
/**
 * @brief In-place matrix subtraction (`mat1 -= mat2`).
 * @param mat1 Matrix to subtract from.
 * @param mat2 Matrix to subtract (must have same shape as mat1).
 * @return true on success, false on shape/allocation error.
 */
bool sm_inplace_diff(FloatMatrix *mat1, const FloatMatrix *mat2);
/**
 * @brief In-place transpose for square matrices.
 * @param mat Square matrix to transpose.
 * @return true on success, false if matrix is not square.
 */
bool sm_inplace_square_transpose(FloatMatrix *mat);
/**
 * @brief In-place scalar multiplication.
 * @param mat Matrix to scale.
 * @param scalar Multiplicative factor.
 * @return true on success, false on error.
 */
bool sm_inplace_multiply_by_number(FloatMatrix *mat, const float scalar);
/**
 * @brief In-place element-wise multiplication.
 * @param mat1 Matrix to scale element-wise.
 * @param mat2 Element-wise scale matrix (must have same shape as mat1).
 * @return true on success, false on shape/allocation error.
 */
bool sm_inplace_elementwise_multiply(FloatMatrix *mat1,
                                     const FloatMatrix *mat2);
/**
 * @brief In-place element-wise division.
 * @param mat1 Numerator matrix to divide.
 * @param mat2 Denominator matrix (must have same shape as mat1).
 * @return true on success, false on shape/zero-division error.
 */
bool sm_inplace_div(FloatMatrix *mat1, const FloatMatrix *mat2);
/**
 * @brief In-place row-wise normalization (L2 norm).
 * @param mat Matrix to normalize (each row becomes unit length).
 * @return true on success, false on error.
 */
bool sm_inplace_normalize_rows(FloatMatrix *mat);
/**
 * @brief In-place column-wise normalization (L2 norm).
 * @param mat Matrix to normalize (each column becomes unit length).
 * @return true on success, false on error.
 */
bool sm_inplace_normalize_cols(FloatMatrix *mat);

/**************************************/
/*       Matrix Properties            */
/**************************************/
/**
 * @brief Determinant of a square matrix.
 * @param mat Source matrix (must be square).
 * @return Determinant value, or NaN on error/singular matrix.
 */
float sm_determinant(const FloatMatrix *mat);
/**
 * @brief Trace of a square matrix (sum of diagonal elements).
 * @param mat Source matrix (must be square).
 * @return Trace value.
 */
float sm_trace(const FloatMatrix *mat);
/**
 * @brief Matrix norm (Frobenius norm).
 * @param mat Source matrix.
 * @return Frobenius norm (square root of sum of squared elements).
 */
float sm_norm(const FloatMatrix *mat);
/**
 * @brief Matrix rank.
 * @param mat Source matrix.
 * @return Approximate rank (number of non-negligible singular values).
 */
size_t sm_rank(const FloatMatrix *mat);
/**
 * @brief Matrix density in range `[0,1]`.
 * @param mat Source matrix.
 * @return Ratio of non-zero elements to total elements.
 */
float sm_density(const FloatMatrix *mat);

/**************************************/
/*       Matrix Property Checks       */
/**************************************/
/**
 * @brief Check whether matrix is empty/uninitialized.
 * @param mat Source matrix.
 * @retval true Matrix has NULL values pointer.
 * @retval false Matrix has allocated storage.
 */
bool sm_is_empty(const FloatMatrix *mat);
/**
 * @brief Check whether matrix is square.
 * @param mat Source matrix.
 * @retval true Matrix has rows == cols.
 * @retval false Matrix is rectangular.
 */
bool sm_is_square(const FloatMatrix *mat);
/**
 * @brief Check whether matrix represents a vector.
 * @param mat Source matrix.
 * @retval true Matrix has shape 1 x n or n x 1.
 * @retval false Matrix is 2D and not a vector.
 */
bool sm_is_vector(const FloatMatrix *mat);
/**
 * @brief Check whether two matrices have equal shape.
 * @param mat1 First matrix.
 * @param mat2 Second matrix.
 * @retval true Both matrices have same rows and cols.
 * @retval false Matrices have different shapes.
 */
bool sm_is_equal_size(const FloatMatrix *mat1, const FloatMatrix *mat2);
/**
 * @brief Check whether two matrices are element-wise equal.
 * @param mat1 First matrix.
 * @param mat2 Second matrix (must have same shape as mat1).
 * @retval true All corresponding elements are equal.
 * @retval false Matrices differ or have incompatible shapes.
 */
bool sm_is_equal(const FloatMatrix *mat1, const FloatMatrix *mat2);
/**
 * @brief Perform LU decomposition (in-place); writes pivot order.
 * @param mat Matrix to decompose (overwritten with LU result).
 * @param pivot_order Array of length rows to store pivot indices.
 * @return true on success, false on allocation/singular matrix error.
 */
bool sm_lu_decompose(FloatMatrix *mat, size_t *pivot_order);

/**************************************/
/*         Matrix Utilities           */
/**************************************/

/**
 * @brief Print matrix to stdout (debug helper).
 * @param matrix Matrix to print.
 */
void sm_print(const FloatMatrix *matrix);

/**
 * @brief Return active compute backend name.
 * @return String name of active backend (e.g., "Accelerate", "MPS", "OpenBLAS").
 */
const char *sm_active_library(void);

/**
 * @brief Return whether MPS backend is available in this build.
 * @retval true MPS (Metal) backend is available.
 * @retval false MPS backend is not available.
 */
bool sm_mps_available(void);

/**************************************/
/*         Memory Management          */
/**************************************/

/**
 * @brief Destroy matrix and release all allocated memory.
 * @param mat Matrix pointer (NULL-safe; no-op if NULL).
 */
void sm_destroy(FloatMatrix *mat);

#endif  // SM_H
