/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef DM_H
#define DM_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/**************************************/
/*         Matrix Struct              */
/**************************************/
typedef struct DoubleMatrix {
  size_t rows;
  size_t cols;
  size_t capacity;
  double *values;
} DoubleMatrix;

/**************************************/
/*         Matrix Creation            */
/**************************************/
DoubleMatrix *dm_create_empty(void);
DoubleMatrix *dm_create_with_values(size_t rows, size_t cols, double *values);
DoubleMatrix *dm_create(size_t rows, size_t cols);
DoubleMatrix *dm_create_clone(const DoubleMatrix *m);
DoubleMatrix *dm_create_identity(size_t n);
DoubleMatrix *dm_create_random(size_t rows, size_t cols);

/**************************************/
/*         Matrix Import              */
/**************************************/
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols, double **array);
DoubleMatrix *dm_create_from_2D_array(size_t rows, size_t cols,
                                      double array[rows][cols]);

/**************************************/
/*         Matrix Accessors           */
/**************************************/
double dm_get(const DoubleMatrix *mat, size_t i, size_t j);
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
void dm_destroy(DoubleMatrix *mat);

#endif // DM_H
