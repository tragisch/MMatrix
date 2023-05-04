#ifndef MATRIX_UR_H
#define MATRIX_UR_H

#include "misc.h"
#include <stdbool.h>

// #define NDEBUG

/*******************************/
/*     Define & Types          */
/*******************************/

// sparse matrix formats
typedef enum { DENSE, SPARSE } matrix_format;

// Standard is of SparseMatrix (COO-Format)
typedef struct DoubleMatrix {
  size_t rows;
  size_t cols;
  size_t row_capacity;  // Capacity of row pointers
  size_t col_capacity;  // Capacity of column indices
  size_t nnz;           // Number of non-zero elements
  size_t *row_pointers; // Array of row pointers
  size_t *col_indices;  // Array of column indices of non-zero elements
  matrix_format format; // COO, CSC, CSR
  double *values;       // Values
} DoubleMatrix;

// Definition of DoubleVector
typedef DoubleMatrix DoubleVector;

/*******************************/
/*     Create  & convert       */
/*******************************/

// Create, Clone, Destroy
DoubleMatrix *dm_create(size_t rows, size_t cols); // empty sparse matrix
DoubleMatrix *dm_create_format(size_t rows, size_t cols, matrix_format format);

DoubleMatrix *dm_create_sparse(size_t rows, size_t cols);
DoubleMatrix *dm_create_dense(size_t rows, size_t cols);

// convert:
void dm_convert(DoubleMatrix *mat, matrix_format format);
void dm_convert_to_sparse(DoubleMatrix *mat);
void dm_convert_to_dense(DoubleMatrix *mat);

/*******************************/
/*        Getter & Setter      */
/*******************************/

double dm_get(const DoubleMatrix *mat, size_t i, size_t j);
double dm_get_dense(const DoubleMatrix *mat, size_t i, size_t j);
double dm_get_sparse(const DoubleMatrix *mat, size_t i, size_t j);

void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value);
void dm_set_sparse(DoubleMatrix *mat, size_t i, size_t j, double value);
void dm_set_dense(DoubleMatrix *mat, size_t i, size_t j, double value);

double dm_get(const DoubleMatrix *mat, size_t i, size_t j);
void dm_set(DoubleMatrix *mat, size_t i, size_t j, const double value);

/*******************************/
/*       Resize / Parts        */
/*******************************/

// shrink, push, pop, expand
void dm_resize(DoubleMatrix *mat, size_t rows, size_t cols);
void dm_resize_dense(DoubleMatrix *mat, size_t rows, size_t cols);
void dm_resize_sparse(DoubleMatrix *mat, size_t new_rows, size_t new_cols);
static void compute_new_row_pointers_and_nnz(DoubleMatrix *mat, size_t new_rows,
                                             size_t *new_nnz,
                                             size_t *new_row_capacity);
static size_t compute_new_col_capacity(DoubleMatrix *mat, size_t new_cols);
static void resize_col_arrays(DoubleMatrix *mat, size_t new_col_capacity);

/*******************************/
/*    Rand, Clone, Identity    */
/*******************************/

DoubleMatrix *dm_create_rand(size_t rows, size_t cols, double density);
DoubleMatrix *dm_clone(DoubleMatrix *m);
DoubleMatrix *dm_create_identity(size_t rows);
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols,
                                   double array[rows][cols]);
DoubleMatrix *dm_get_sub_matrix(DoubleMatrix *mat, size_t row_start,
                                size_t row_end, size_t col_start,
                                size_t col_end);

// static getter

// Test if vector or matrix (true = vector)
bool dm_is_vector(DoubleMatrix *mat);

// Getters and Setters
void dm_push_column(DoubleMatrix *mat, DoubleVector *col_vec);
void dm_push_row(DoubleMatrix *mat, DoubleVector *row_vec);

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
DoubleMatrix *sp_laplace(size_t n);
DoubleMatrix *sp_hilbert(size_t n);
DoubleMatrix *sp_vandermonde(size_t n);
DoubleMatrix *sp_toeplitz(size_t n);
DoubleMatrix *sp_circulant(size_t n);
DoubleMatrix *sp_hankel(size_t n);
DoubleMatrix *sp_dft(size_t n);
DoubleMatrix *sp_dct(size_t n);
DoubleMatrix *sp_dst(size_t n);
DoubleMatrix *sp_hadamard(size_t n);
DoubleMatrix *sp_kronecker(size_t n);
DoubleMatrix *sp_toeplitz(size_t n);
DoubleMatrix *sp_adjacent(size_t n);

// transformations
void sp_to_laplace(DoubleMatrix *A);

#endif // !MATRIX_H
