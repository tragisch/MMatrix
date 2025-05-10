/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef SM_H
#define SM_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Struct for FloatMatrix
typedef struct FloatMatrix {
  size_t rows;
  size_t cols;
  size_t capacity;
  float *values;
} FloatMatrix;

// Function declarations
FloatMatrix *sm_create_empty();
FloatMatrix *sm_create_with_values(size_t rows, size_t cols, float *values);
FloatMatrix *sm_create(size_t rows, size_t cols);
FloatMatrix *sm_create_clone(const FloatMatrix *m);
FloatMatrix *sm_create_identity(size_t n);
FloatMatrix *sm_create_random(size_t rows, size_t cols);
FloatMatrix *sm_create_random_he(size_t rows, size_t cols, size_t fan_in);
FloatMatrix *sm_create_random_xavier(size_t rows, size_t cols, size_t fan_in,
                                     size_t fan_out);

// Importing from array
FloatMatrix *sm_create_from_array(size_t rows, size_t cols, float **array);
FloatMatrix *sm_create_from_2D_array(size_t rows, size_t cols,
                                     float array[rows][cols]);

double *sm_create_array_from_matrix(FloatMatrix *matrix);

// Getters and Setters
float sm_get(const FloatMatrix *mat, size_t i, size_t j);
void sm_set(FloatMatrix *mat, size_t i, size_t j, float value);

FloatMatrix *sm_get_row(const FloatMatrix *mat, size_t i);
FloatMatrix *sm_get_last_row(const FloatMatrix *mat);
FloatMatrix *sm_get_col(const FloatMatrix *mat, size_t j);
FloatMatrix *sm_get_last_col(const FloatMatrix *mat);

// Matrix slicing and reshaping
void sm_reshape(FloatMatrix *matrix, size_t new_rows, size_t new_cols);
void sm_resize(FloatMatrix *mat, size_t new_row, size_t new_col);

// Matrix operations
FloatMatrix *sm_multiply(const FloatMatrix *mat1, const FloatMatrix *mat2);
FloatMatrix *sm_multiply_by_number(const FloatMatrix *mat, const float number);
FloatMatrix *sm_transpose(const FloatMatrix *mat);
FloatMatrix *sm_add(const FloatMatrix *mat1, const FloatMatrix *mat2);
FloatMatrix *sm_diff(const FloatMatrix *mat1, const FloatMatrix *mat2);
FloatMatrix *sm_inverse(const FloatMatrix *mat);

// Matrix properties
float sm_determinant(const FloatMatrix *mat);
float sm_trace(const FloatMatrix *mat);
size_t sm_rank(const FloatMatrix *mat);
float sm_norm(const FloatMatrix *mat);
float sm_density(const FloatMatrix *mat);

// Matrix properties (boolean)
bool sm_is_empty(const FloatMatrix *mat);
bool sm_is_square(const FloatMatrix *mat);
bool sm_is_vector(const FloatMatrix *mat);
bool sm_is_equal_size(const FloatMatrix *mat1, const FloatMatrix *mat2);
bool sm_is_equal(const FloatMatrix *mat1, const FloatMatrix *mat2);

// In-place operations
void sm_inplace_add(FloatMatrix *mat1, const FloatMatrix *mat2);
void sm_inplace_diff(FloatMatrix *mat1, const FloatMatrix *mat2);
void sm_inplace_square_transpose(FloatMatrix *mat);
void sm_inplace_multiply_by_number(FloatMatrix *mat, const float scalar);
void sm_inplace_gauss_elimination(FloatMatrix *mat);

// Neural network operations
FloatMatrix *sm_linear_batch(const FloatMatrix *inputs,
                             const FloatMatrix *weights,
                             const FloatMatrix *biases);
// Memory management
void sm_destroy(FloatMatrix *mat);

// File I/O
void sm_print(const FloatMatrix *matrix);
char *sm_active_library();

#endif // sm_H
