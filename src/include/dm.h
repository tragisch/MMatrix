/*
 * dm.h - Double Matrix Library
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
#include <matio.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INIT_CAPACITY 100
#define EPSILON 1e-10

// Macro for measuring execution time of a function call
#define CPUTIME(FCALL)                                                         \
  do {                                                                         \
    double START = clock();                                                    \
    FCALL;                                                                     \
    ((double)clock() - START) / CLOCKS_PER_SEC;                                \
  } while (0)

// Struct for DoubleMatrix
typedef struct {
  size_t rows;
  size_t cols;
  size_t capacity;
  double *values;
} DoubleMatrix;

// Function declarations

// Matrix creation and initialization
DoubleMatrix *dm_create(size_t rows, size_t cols);
DoubleMatrix *dm_clone(const DoubleMatrix *m);
DoubleMatrix *dm_identity(size_t n);
DoubleMatrix *dm_rand(size_t rows, size_t cols, double density);

// Importing from array
DoubleMatrix *dm_import_array(size_t rows, size_t cols, double **array);
DoubleMatrix *dm_convert_array(size_t rows, size_t cols,
                               double array[rows][cols]);

// Matrix slicing and reshaping
DoubleMatrix *dm_get_row(const DoubleMatrix *mat, size_t i);
DoubleMatrix *dm_get_last_row(const DoubleMatrix *mat);
DoubleMatrix *dm_get_col(const DoubleMatrix *mat, size_t j);
DoubleMatrix *dm_get_last_col(const DoubleMatrix *mat);
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
double dm_determinant(const DoubleMatrix *mat);
double dm_trace(const DoubleMatrix *mat);
size_t dm_rank(const DoubleMatrix *mat);
double dm_norm(const DoubleMatrix *mat);

// In-place operations
void dm_multiply_me_by_number(DoubleMatrix *mat, const double scalar);

// Utility functions
bool dm_equal(const DoubleMatrix *mat1, const DoubleMatrix *mat2);
double dm_get(const DoubleMatrix *mat, size_t i, size_t j);
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value);
void dm_destroy(DoubleMatrix *mat);
void dm_print(const DoubleMatrix *matrix);

// File I/O with Matio library
int dm_write_MAT_file(const DoubleMatrix *matrix, const char *filename);
DoubleMatrix *dm_read_MAT_file(const char *filename, const char *varname);

// Private functions (should not be used outside this file)
static size_t dm_rank_euler(const DoubleMatrix *mat);
static void dm_gauss_elimination(DoubleMatrix *mat);

#endif // DM_H
