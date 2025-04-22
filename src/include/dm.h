/*
 * dm_create_empty.h - Double Matrix Library
 * author: uwe@roettgermann.de
 * for my private purpose: a simple (dense) matrix library using doubles
 * On MacOS it uses Apples Accelerate framework, BLAS
 *
 * License: MIT
 * Last modified: 2024-06-01
 * (c) 2021 Uwe Röttgermann
 */

#ifndef DM_H
#define DM_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Macro for measuring execution time of a function call
#define CPUTIME(FCALL)                                                         \
  do {                                                                         \
    double START = clock();                                                    \
    FCALL;                                                                     \
    ((double)clock() - START) / CLOCKS_PER_SEC;                                \
  } while (0)

// Struct for DoubleMatrix
typedef struct DoubleMatrix {
  size_t rows;
  size_t cols;
  size_t capacity;
  double *values;
} DoubleMatrix;

// Function declarations
DoubleMatrix *dm_create_empty();
DoubleMatrix *dm_create_with_values(size_t rows, size_t cols, double *values);
DoubleMatrix *dm_create(size_t rows, size_t cols);
DoubleMatrix *dm_create_clone(const DoubleMatrix *m);
DoubleMatrix *dm_create_identity(size_t n);
DoubleMatrix *dm_create_random(size_t rows, size_t cols);

// Importing from array
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols, double **array);
DoubleMatrix *dm_create_from_2D_array(size_t rows, size_t cols,
                                      double array[rows][cols]);

// Getters and Setters
double dm_get(const DoubleMatrix *mat, size_t i, size_t j);
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value);

DoubleMatrix *dm_get_row(const DoubleMatrix *mat, size_t i);
DoubleMatrix *dm_get_last_row(const DoubleMatrix *mat);
DoubleMatrix *dm_get_col(const DoubleMatrix *mat, size_t j);
DoubleMatrix *dm_get_last_col(const DoubleMatrix *mat);

// Matrix slicing and reshaping
void dm_reshape(DoubleMatrix *matrix, size_t new_rows, size_t new_cols);
void dm_resize(DoubleMatrix *mat, size_t new_row, size_t new_col);

// Matrix operations
DoubleMatrix *dm_multiply(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
DoubleMatrix *dm_multiply_by_number(const DoubleMatrix *mat,
                                    const double number);
DoubleMatrix *dm_transpose(const DoubleMatrix *mat);
DoubleMatrix *dm_add(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
DoubleMatrix *dm_diff(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
DoubleMatrix *dm_inverse(const DoubleMatrix *mat);

// Matrix properties
double dm_determinant(const DoubleMatrix *mat);
double dm_trace(const DoubleMatrix *mat);
size_t dm_rank(const DoubleMatrix *mat);
double dm_norm(const DoubleMatrix *mat);
double dm_density(const DoubleMatrix *mat);

// Matrix properties (boolean)
bool dm_is_empty(const DoubleMatrix *mat);
bool dm_is_square(const DoubleMatrix *mat);
bool dm_is_vector(const DoubleMatrix *mat);
bool dm_is_equal_size(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
bool dm_is_equal(const DoubleMatrix *mat1, const DoubleMatrix *mat2);

// In-place operations
void dm_inplace_add(DoubleMatrix *mat1, const DoubleMatrix *mat2);
void dm_inplace_diff(DoubleMatrix *mat1, const DoubleMatrix *mat2);
void dm_inplace_transpose(DoubleMatrix *mat);
void dm_inplace_multiply_by_number(DoubleMatrix *mat, const double scalar);
void dm_inplace_gauss_elimination(DoubleMatrix *mat);

// Memory management
void dm_destroy(DoubleMatrix *mat);

// File I/O
void dm_print(const DoubleMatrix *matrix);

#endif // DM_H
