/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef DMMa_SPARSE_H
#define DMMa_SPARSE_H

#include <cs.h> // SuiteSparse: a Common Sparse Matrix Package
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// struct of DoubleMatrix
typedef struct DoubleSparseMatrix {
  size_t rows;
  size_t cols;
  size_t nnz;
  size_t capacity;
  size_t *row_indices;
  size_t *col_indices;
  double *values;
} DoubleSparseMatrix;

// Function declarations
DoubleSparseMatrix *dms_create_empty();
DoubleSparseMatrix *dms_create_with_values(size_t rows, size_t cols, size_t nnz,
                                           size_t *row_indices,
                                           size_t *col_indices, double *values);
DoubleSparseMatrix *dms_create(size_t rows, size_t cols, size_t capacity);
DoubleSparseMatrix *dms_create_clone(const DoubleSparseMatrix *m);
DoubleSparseMatrix *dms_create_identity(size_t n);
DoubleSparseMatrix *dms_create_random(size_t rows, size_t cols, double density);

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
void dms_set(DoubleSparseMatrix *mat, size_t i, size_t j, double value);
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
void dms_print(const DoubleSparseMatrix *mat);

// Memory management
void dms_realloc(DoubleSparseMatrix *mat, size_t new_capacity);
void dms_destroy(DoubleSparseMatrix *mat);

#endif // DMMa_SPARSE_H
