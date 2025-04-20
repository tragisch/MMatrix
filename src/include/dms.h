/*
 * dms.h - Double Sparse Matrix Library
 * author: uwe@roettgermann.de
 * for my private purpose: a simple sparse matrix library using doubles
 * On some platforms it uses SuiteSparse: a Common Sparse Matrix Package
 * DoubleSparseMatrix is a sparse matrix in coordinate format (COO).
 *
 * License: MIT
 * Last modified: 2024-06-01
 * (c) 2021 Uwe Röttgermann
 */

#ifndef DMMa_SPARSE_H
#define DMMa_SPARSE_H

#include <cs.h> // SuiteSparse: a Common Sparse Matrix Package
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#ifndef INIT_CAPACITY
#define INIT_CAPACITY 100
#endif
#ifndef EPSILON
#define EPSILON 1e-10
#endif

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

#endif // DMMa_SPARSE_H
DoubleSparseMatrix *dms();
DoubleSparseMatrix *dms_create_test_matrix(size_t rows, size_t cols, size_t nnz,
                                           size_t *row_indices,
                                           size_t *col_indices, double *values);
DoubleSparseMatrix *dms_create(size_t rows, size_t cols, size_t capacity);
DoubleSparseMatrix *dms_clone(const DoubleSparseMatrix *m);
DoubleSparseMatrix *dms_identity(size_t n);
DoubleSparseMatrix *dms_rand(size_t rows, size_t cols, double density);

// Converting to cs-sparse format
cs *dms_to_cs(const DoubleSparseMatrix *coo);
DoubleSparseMatrix *cs_to_dms(const cs *A);

double *dms_to_array(const DoubleSparseMatrix *mat);
DoubleSparseMatrix *dms_array(size_t rows, size_t cols, double *array);
DoubleSparseMatrix *dms_2D_array(size_t rows, size_t cols,
                                 double array[rows][cols]);

DoubleSparseMatrix *dms_get_row(const DoubleSparseMatrix *mat, size_t i);
DoubleSparseMatrix *dms_get_last_row(const DoubleSparseMatrix *mat);
DoubleSparseMatrix *dms_get_col(const DoubleSparseMatrix *mat, size_t j);
DoubleSparseMatrix *dms_get_last_col(const DoubleSparseMatrix *mat);

DoubleSparseMatrix *dms_multiply(const DoubleSparseMatrix *mat1,
                                 const DoubleSparseMatrix *mat2);
DoubleSparseMatrix *dms_multiply_by_number(const DoubleSparseMatrix *mat,
                                           const double number);
DoubleSparseMatrix *dms_transpose(const DoubleSparseMatrix *mat);

void dms_print(const DoubleSparseMatrix *mat);
void dms_realloc(DoubleSparseMatrix *mat, size_t new_capacity);
double dms_max_double(double a, double b);
double dms_min_double(double a, double b);
int dms_max_int(int a, int b);
void dms_destroy(DoubleSparseMatrix *mat);
void dms_set(DoubleSparseMatrix *mat, size_t i, size_t j, double value);
double dms_get(const DoubleSparseMatrix *mat, size_t i, size_t j);
double dms_density(const DoubleSparseMatrix *mat);

// private functions

static size_t __dms_binary_search(const DoubleSparseMatrix *matrix, size_t i,
                                  size_t j);
static void __dms_insert_element(DoubleSparseMatrix *matrix, size_t i, size_t j,
                                 double value, size_t position);
