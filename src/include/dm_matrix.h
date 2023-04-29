#ifndef MATRIX_UR_H
#define MATRIX_UR_H

#include <stdbool.h>

#include "dv_vector.h"
#include "misc.h"
#include "sp_matrix.h"

// #define NDEBUG

/*******************************/
/*     Define & Types          */
/*******************************/

// sparse matrix formats
typedef enum { COO, CSR, CSC, DENSE } matrix_format;

// Standard is of SparseMatrix (COO-Format)
typedef struct SparseMatrix {
  size_t rows;
  size_t cols;
  size_t nnz;           // Number of non-zero elements
  size_t *row_indices;  // Array of row pointers
  size_t *col_indices;  // Array of column indices of non-zero elements
  matrix_format format; // COO, CSC, CSR
  double *values;       // Values
} SparseMatrix;

// Definition of DoubleVector
typedef SparseMatrix DoubleMatrix;
typedef SparseMatrix DoubleVector;

/*******************************/
/*      Sparse Matrix          */
/*******************************/

SparseMatrix *sp_create(size_t rows, size_t cols);
SparseMatrix *sp_create_format(size_t rows, size_t cols, matrix_format format);
SparseMatrix *sp_create_rand(size_t rows, size_t cols, double density);
SparseMatrix *sp_create_from_array(size_t rows, size_t cols, size_t nnz,
                                   size_t *row_indices, size_t *col_indices,
                                   double *values);

void sp_convert_to_csc(SparseMatrix *mat);
void sp_convert_to_csr(SparseMatrix *mat);
void sp_convert_to_coo(SparseMatrix *mat);
void sp_convert_to_dense(SparseMatrix *mat);
void sp_convert_to_sparse(DoubleMatrix *mat);

bool sp_is_valid(const SparseMatrix *a);
void sp_destroy(SparseMatrix *sp_matrix);

double sp_get(const SparseMatrix *mat, size_t i, size_t j);
void sp_set(SparseMatrix *mat, size_t i, size_t j, double value);

/*******************************/
/*        Double Matrix        */
/*******************************/

// Create, Clone, Destroy
DoubleMatrix *dm_matrix();
DoubleMatrix *dm_create(size_t rows, size_t cols);
DoubleMatrix *dm_create_rand(size_t rows, size_t cols);
DoubleMatrix *dm_clone(DoubleMatrix *m);
DoubleMatrix *dm_create_identity(size_t rows);
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols,
                                   double array[rows][cols]);
DoubleMatrix *dm_get_sub_matrix(DoubleMatrix *mat, size_t row_start,
                                size_t row_end, size_t col_start,
                                size_t col_end);

// Test if vector or matrix (true = vector)
bool dm_is_vector(DoubleMatrix *mat);

// shrink, push, pop, expand
void dm_resize(DoubleMatrix *mat, size_t rows, size_t cols);

// Getters and Setters
void dm_push_column(DoubleMatrix *mat, DoubleVector *col_vec);
void dm_push_row(DoubleMatrix *mat, DoubleVector *row_vec);

double dm_get(const DoubleMatrix *mat, size_t i, size_t j);
void dm_set(DoubleMatrix *mat, size_t i, size_t j, const double value);

// free
void dm_destroy(DoubleMatrix *mat);

/*******************************/
/*  Double Vector  (Dynamic)   */
/*******************************/

// Create, Clone, Destroy
DoubleVector *dv_vector();
DoubleVector *dv_create(size_t length);
DoubleVector *dv_create_rand(size_t length);
DoubleVector *dv_create_from_array(const double *array, const size_t length);
DoubleVector *dv_clone(DoubleVector *vector);

// Test if vector is a column or row vector:
bool dv_is_row_vector(const DoubleVector *vec);

// Get DoubleVector from DoubleMatrix:
DoubleVector *dv_get_row_vector(DoubleMatrix *mat, size_t row);
DoubleVector *dv_get_column_vector(DoubleMatrix *mat, size_t column);
DoubleVector *dm_pop_column_vector(DoubleMatrix *mat);
DoubleVector *dm_pop_row_vector(DoubleMatrix *mat);

// Getters and Setters
double *dv_get_array(const DoubleVector *vec);
void dv_set(DoubleVector *vec, size_t idx, double value);
double dv_get(const DoubleVector *vec, size_t idx);
void dv_push_value(DoubleVector *vec, double value);
double dv_pop_value(DoubleVector *vec);

// shrink, push, pop, expand
void dv_resize(DoubleVector *vec, size_t rows);

// free
void dv_destroy(DoubleVector *vec);

/*******************************/
/* Special quadratic Matrizes  */
/*******************************/

// help from github copilot

// standards
SparseMatrix *sp_laplace(size_t n);
SparseMatrix *sp_hilbert(size_t n);
SparseMatrix *sp_vandermonde(size_t n);
SparseMatrix *sp_toeplitz(size_t n);
SparseMatrix *sp_circulant(size_t n);
SparseMatrix *sp_hankel(size_t n);
SparseMatrix *sp_dft(size_t n);
SparseMatrix *sp_dct(size_t n);
SparseMatrix *sp_dst(size_t n);
SparseMatrix *sp_hadamard(size_t n);
SparseMatrix *sp_kronecker(size_t n);
SparseMatrix *sp_toeplitz(size_t n);
SparseMatrix *sp_adjacent(size_t n);

// transformations
void sp_to_laplace(SparseMatrix *A);

#endif // !MATRIX_H
