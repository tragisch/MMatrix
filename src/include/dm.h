#ifndef DM_H
#define DM_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/*******************************/
/*     Define & Types          */
/*******************************/

enum { INIT_CAPACITY = 100U };

// sparse matrix formats
typedef enum { DENSE, COO, CSC, VECTOR } matrix_format;

extern matrix_format default_matrix_format;

// struct of DoubleMatrix
typedef struct DoubleMatrix {
  size_t rows;          // Number of rows
  size_t cols;          // Number of columns
  size_t capacity;      // Capacity of row_indices and col_indices
  size_t nnz;           // Number of non-zero elements
  size_t *row_indices;  // COO: Array of row indices of non-zero elements,
  size_t *col_ptrs;     // CSR: Array of row pointers
  size_t *col_indices;  // Array of column indices of non-zero elements
  matrix_format format; // COO, CSC, DENSE or VECTOR
  double *values;       // Values
} DoubleMatrix;

// Definition of DoubleVector
typedef DoubleMatrix DoubleVector;

/*******************************/
/*     Create  & convert       */
/*******************************/

// Create, Clone, Destroy
void set_default_matrix_format(matrix_format format);
DoubleMatrix *dm_create(size_t rows, size_t cols); // empty sparse matrix
DoubleMatrix *dm_create_nnz(size_t rows, size_t cols, size_t nnz);
DoubleMatrix *dm_create_format(size_t rows, size_t cols, matrix_format format);
DoubleMatrix *dm_clone(const DoubleMatrix *m);

// free memory
void dm_destroy(DoubleMatrix *mat);

/*******************************/
/*        Getter & Setter      */
/*******************************/

double dm_get(const DoubleMatrix *mat, size_t i, size_t j);
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value);
size_t binary_search_coo(const DoubleMatrix *matrix, size_t i, size_t j);

void dm_realloc_csc(DoubleMatrix *mat, size_t new_capacity);
void dm_realloc_coo(DoubleMatrix *mat, size_t new_capacity);

/*******************************/
/*     Special Matrices        */
/*******************************/

DoubleMatrix *dm_create_rand(size_t rows, size_t cols, double density);
DoubleMatrix *dm_create_rand_between(size_t rows, size_t cols, size_t min,
                                     size_t max, double density);
DoubleMatrix *dm_create_identity(size_t rows);
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols,
                                   double array[rows][cols]);

DoubleMatrix *dm_create_diagonal(size_t rows, size_t cols, double array[rows]);

#endif // DM_H
