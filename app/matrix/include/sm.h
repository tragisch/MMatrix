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

/**************************************/
/*         Float Matrix Struct.       */
/**************************************/

/**
 * @struct FloatMatrix
 * @brief Dense matrix of single-precision (float) values with row-major storage.
 *
 * TENSOR SEMANTICS (Identical to DoubleMatrix, but with 32-bit floats):
 *   - Storage Format: Row-major (C-contiguous)
 *   - Index Formula: values[i * cols + j] for element at (i, j)
 *   - Capacity: Always equals rows * cols (dense, no gaps)
 *   - Precision: IEEE 754 single (32-bit float), ~1e-7 relative error
 *
 * NEURAL NETWORK OPTIMIZED:
 *   - 2× less memory than DoubleMatrix (32-bit vs. 64-bit)
 *   - 2-10× faster on GPU (CUDA/Metal native float ops)
 *   - Industry standard (TensorFlow, PyTorch, ONNX all use float32)
 *   - Sufficient precision for training/inference (~1e-7 >> 1e-4 gradient scales)
 *
 * MEMORY OWNERSHIP & THREAD-SAFETY:
 *   (Same as DoubleMatrix)
 *   - Caller owns struct + values array
 *   - Not thread-safe: concurrent writes to same cell are race conditions
 *   - dm_set_random_seed() modifies global state; serialize access
 *   - dm_get() is safe if no concurrent writes
 *
 * @see DoubleMatrix (dm.h) for detailed semantics
 */
typedef struct FloatMatrix {
  size_t rows;  /**< Number of rows. */
  size_t cols;  /**< Number of columns. */
  size_t capacity;  /**< Total allocated elements (= rows * cols for dense). */
  float *values;  /**< Row-major array of 32-bit floats. values[i*cols+j] = mat[i][j] */
} FloatMatrix;

/**************************************/
/*         Matrix Creation            */
/**************************************/

/**
 * @brief Create an empty float matrix (NULL values pointer).
 *
 * @return Pointer to FloatMatrix, or NULL if malloc fails.
 * @see sm_create, sm_destroy
 */
FloatMatrix *sm_create_empty(void);

/**
 * @brief Create float matrix initialized to zero.
 *
 * SEMANTICS:
 *   - Allocates struct + values array
 *   - All values = 0.0f (not uninitialized; explicit zeroing)
 *   - Useful for accumulation, bias initialization
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 *
 * @return Zero-initialized FloatMatrix, or NULL if allocation failed.
 *
 * @see sm_create, sm_destroy
 */
FloatMatrix *sm_create_zeros(size_t rows, size_t cols);

/**
 * @brief Create uninitialized float matrix.
 *
 * SEMANTICS:
 *   - Like dm_create(), but for float values
 *   - Values are UNINITIALIZED (contain garbage)
 *   - Caller must initialize after creation
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 *
 * @return Uninitialized FloatMatrix, or NULL if allocation failed.
 *
 * @see sm_destroy, sm_create_zeros
 */
FloatMatrix *sm_create(size_t rows, size_t cols);

/**
 * @brief Create float matrix from existing values array (stores pointer).
 *
 * ⚠️ SAME OWNERSHIP RISK AS dm_create_with_values():
 *   Function stores POINTER to values; does NOT copy.
 *   Caller retains ownership of array.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param values Pointer to pre-allocated array of size rows*cols.
 *
 * @return FloatMatrix, or NULL if struct allocation failed.
 *
 * @see sm_destroy, sm_create
 */
FloatMatrix *sm_create_with_values(size_t rows, size_t cols, float *values);

/**
 * @brief Create deep copy of float matrix.
 *
 * @param m Source matrix (const).
 *
 * @return New FloatMatrix (independent copy), or NULL if allocation failed.
 *
 * @see sm_destroy
 */
FloatMatrix *sm_clone(const FloatMatrix *m);

/**
 * @brief Create identity matrix (1s on diagonal, 0s elsewhere).
 *
 * @param n Size of identity matrix (n×n).
 *
 * @return n×n identity matrix with float values, or NULL if allocation failed.
 *
 * @see sm_destroy
 */
FloatMatrix *sm_create_identity(size_t n);

/**
 * @brief Create matrix with random values (uniform [0, 1)) using global seed.
 *
 * SEMANTICS:
 *   - Uses global random seed (set via sm_set_random_seed)
 *   - Non-deterministic by default
 *   - For reproducible NN initialization, use sm_create_random_he() or _xavier()
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (modifies global RNG state)
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 *
 * @return FloatMatrix with random values, or NULL if allocation failed.
 *
 * @see sm_create_random_seeded, sm_create_random_he, sm_create_random_xavier
 */
FloatMatrix *sm_create_random(size_t rows, size_t cols);

/**
 * @brief Create matrix with random values using explicit seed (deterministic, thread-safe).
 *
 * SEMANTICS:
 *   - Same seed → Same matrix output (reproducible)
 *   - Uses SplitMix64 mixing for deterministic generation
 *   - For neural networks with seeded randomness, prefer He/Xavier initializers
 *
 * THREAD-SAFETY:
 *   Thread-safe (no global state modified)
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param seed Seed value. Same seed → same output.
 *
 * @return FloatMatrix with deterministic random values, or NULL if allocation failed.
 *
 * @see sm_set_random_seed, sm_create_random_he, sm_create_random_xavier
 */
FloatMatrix *sm_create_random_seeded(size_t rows, size_t cols, uint64_t seed);

/**
 * @brief Set global random seed for sm_create_random() and NN initializers.
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (modifies global state)
 *   Serialize concurrent seed setting or use _seeded variants
 *
 * @param seed Seed value for global RNG.
 *
 * @see sm_get_random_seed, sm_create_random, sm_create_random_he, sm_create_random_xavier
 */
void sm_set_random_seed(uint64_t seed);

/**
 * @brief Get current global random seed.
 *
 * @return Current global seed value.
 *
 * @see sm_set_random_seed
 */
uint64_t sm_get_random_seed(void);

/**
 * @brief Create weight matrix initialized with He distribution.
 *
 * NEURAL NETWORK SEMANTICS (He Initialization):
 *   - Standard initializer for ReLU networks
 *   - Distribution: Uniform[-sqrt(6/fan_in), sqrt(6/fan_in)]
 *   - Variance stabilization: Accounts for ReLU's zero-killing
 *   - Used with: relu, leaky_relu, elu activations
 *   - NOT recommended for: sigmoid, tanh (use Xavier instead)
 *
 * TYPICAL USAGE:
 *   @code
 *     // Layer neurons: 100 → 50
 *     FloatMatrix *w = sm_create_random_he(100, 50, 100);  // fan_in=100
 *     FloatMatrix *b = sm_create_zeros(50, 1);             // bias
 *     // Use w for forward pass; variance is stable through ReLU
 *   @endcode
 *
 * ERROR HANDLING:
 *   Returns NULL if allocation fails. No validation of fan_in > 0
 *   (undefined behavior if fan_in == 0).
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (uses global random seed)
 *   Serialize seed setting or use deterministic seeding before calling
 *
 * @param rows Number of rows (output units).
 * @param cols Number of columns (input units or 1 for single-sample layers).
 * @param fan_in Number of incoming connections (typically == input_dim).
 *               This controls the variance of the distribution.
 *
 * @return FloatMatrix with He-initialized values, or NULL if allocation failed.
 *
 * REFERENCES:
 *   He, K., Zhang, X., Ren, S., & Sun, J. (2015).
 *   "Delving Deep into Rectifiers: Surpassing Human-Level Performance"
 *   https://arxiv.org/abs/1502.01852
 *
 * @see sm_create_random_xavier, sm_set_random_seed
 */
FloatMatrix *sm_create_random_he(size_t rows, size_t cols, size_t fan_in);

/**
 * @brief Create weight matrix initialized with Xavier/Glorot distribution.
 *
 * NEURAL NETWORK SEMANTICS (Xavier/Glorot Initialization):
 *   - Standard initializer for sigmoid/tanh networks
 *   - Distribution: Uniform[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
 *   - Variance stabilization: Balances fan_in and fan_out
 *   - Maintains constant variance through network layers
 *   - Used with: sigmoid, tanh, linear activations
 *   - Also works with ReLU (but He is preferred for ReLU)
 *
 * TYPICAL USAGE:
 *   @code
 *     // Layer neurons: 100 → 50
 *     FloatMatrix *w = sm_create_random_xavier(100, 50, 100, 50);
 *     FloatMatrix *b = sm_create_zeros(50, 1);
 *     // Variance stable in both forward and backward passes
 *   @endcode
 *
 * MATHEMATICS:
 *   Var[W] = 1 / (fan_in + fan_out)
 *   This ensures signal doesn't explode/vanish through sigmoid/tanh
 *
 * ERROR HANDLING:
 *   Returns NULL if allocation fails. No validation of fan_in/fan_out.
 *   (undefined behavior if both == 0).
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (uses global random seed)
 *
 * @param rows Number of rows (output units).
 * @param cols Number of columns (input units or 1).
 * @param fan_in Number of incoming connections.
 * @param fan_out Number of outgoing connections (next layer input size).
 *
 * @return FloatMatrix with Xavier-initialized values, or NULL if allocation failed.
 *
 * REFERENCES:
 *   Glorot, X., & Bengio, Y. (2010).
 *   "Understanding the difficulty of training deep feedforward neural networks"
 *   https://proceedings.mlr.press/v9/glorot10a.html
 *
 * @see sm_create_random_he, sm_set_random_seed
 */
FloatMatrix *sm_create_random_xavier(size_t rows, size_t cols, size_t fan_in,
                                     size_t fan_out);

/**
 * @brief Create matrix from array of pointers (rows may be non-contiguous).
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param array Array of float* (each pointer to a row of cols elements).
 *
 * @return FloatMatrix flattened from jagged array, or NULL if allocation failed.
 *
 * @see sm_from_array_static, sm_destroy
 */
FloatMatrix *sm_from_array_ptrs(size_t rows, size_t cols, float **array);

/**
 * @brief Create matrix from static 2D array (C99 VLA syntax).
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param array Static 2D array array[rows][cols].
 *
 * @return FloatMatrix with copied data, or NULL if allocation failed.
 *
 * @see sm_from_array_ptrs, sm_destroy
 */
FloatMatrix *sm_from_array_static(size_t rows, size_t cols,
                                  float array[rows][cols]);

/**
 * @brief Convert float matrix to 1D C array (row-major).
 *
 * SEMANTICS:
 *   - Allocates new array
 *   - Copies matrix values in row-major order
 *   - Caller owns returned array; must free()
 *
 * USAGE (e.g., export to NumPy/PyTorch):
 *   @code
 *     float *flat = sm_create_array_from_matrix(matrix);
 *     // Send flat to Python via ctypes or similar
 *     free(flat);
 *   @endcode
 *
 * @param matrix Matrix to convert (not modified).
 *
 * @return Pointer to float array of size rows*cols, or NULL if malloc fails.
 *         Caller must free() when done.
 *
 * @see sm_create_with_values
 */
float *sm_create_array_from_matrix(FloatMatrix *matrix);

/**************************************/
/*         Matrix Accessors           */
/**************************************/

/**
 * @brief Get element at (i, j).
 *
 * @param mat Matrix (const).
 * @param i Row index.
 * @param j Column index.
 *
 * @return Element value (float).
 *
 * @see sm_set, DoubleMatrix::dm_get (identical semantics)
 */
float sm_get(const FloatMatrix *mat, size_t i, size_t j);

/**
 * @brief Set element at (i, j).
 *
 * ⚠️ NOT thread-safe if concurrent writes to same cell occur.
 *
 * @param mat Matrix (modified).
 * @param i Row index.
 * @param j Column index.
 * @param value New value.
 *
 * @see sm_get
 */
void sm_set(FloatMatrix *mat, size_t i, size_t j, float value);
FloatMatrix *sm_get_row(const FloatMatrix *mat, size_t i);
FloatMatrix *sm_get_last_row(const FloatMatrix *mat);
FloatMatrix *sm_get_col(const FloatMatrix *mat, size_t j);
FloatMatrix *sm_get_last_col(const FloatMatrix *mat);
FloatMatrix *sm_slice_rows(const FloatMatrix *mat, size_t start, size_t end);

/**************************************/
/*         Matrix Shape Ops           */
/**************************************/
void sm_reshape(FloatMatrix *matrix, size_t new_rows, size_t new_cols);
void sm_resize(FloatMatrix *mat, size_t new_row, size_t new_col);

/**************************************/
/*       Matrix Transformations       */
/**************************************/
FloatMatrix *sm_transpose(const FloatMatrix *mat);

/**************************************/
/*       Matrix Arithmetic            */
/**************************************/
typedef enum SmTranspose {
  SM_NO_TRANSPOSE = 0,
  SM_TRANSPOSE = 1,
} SmTranspose;

bool sm_gemm(FloatMatrix *C, float alpha, const FloatMatrix *A,
             SmTranspose trans_a, const FloatMatrix *B, SmTranspose trans_b,
             float beta);
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
/*        Matrix In-Place Ops         */
/**************************************/
void sm_inplace_add(FloatMatrix *mat1, const FloatMatrix *mat2);
void sm_inplace_diff(FloatMatrix *mat1, const FloatMatrix *mat2);
void sm_inplace_square_transpose(FloatMatrix *mat);
void sm_inplace_multiply_by_number(FloatMatrix *mat, const float scalar);
void sm_inplace_elementwise_multiply(FloatMatrix *mat1,
                                     const FloatMatrix *mat2);
void sm_inplace_div(FloatMatrix *mat1, const FloatMatrix *mat2);
void sm_inplace_normalize_rows(FloatMatrix *mat);
void sm_inplace_normalize_cols(FloatMatrix *mat);

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

/**
 * @brief Print matrix to stdout (for debugging).
 *
 * @param matrix Matrix to print (const).
 *
 * @see dm_print (identical)
 */
void sm_print(const FloatMatrix *matrix);

/**
 * @brief Get name of active matrix library (BLAS, Accelerate, etc.).
 *
 * @return Const string with library name.
 *
 * @see dm_active_library
 */
const char *sm_active_library(void);

/**************************************/
/*         Memory Management          */
/**************************************/

/**
 * @brief Free a float matrix and all associated memory.
 *
 * SEMANTICS:
 *   - Deallocates both struct and values array
 *   - When called on matrix from sm_create_with_values(),
 *     frees struct only (not the values array; caller owns that)
 *   - Safe to call on NULL (no-op)
 *
 * ⚠️ DO NOT call twice on same pointer (double-free crash).
 *
 * @param mat Matrix to deallocate.
 *
 * @see sm_create, sm_create_empty, DoubleMatrix::dm_destroy (identical)
 */
void sm_destroy(FloatMatrix *mat);

#endif  // sm_H
