#ifndef SP_MATRIX_UR_H
#define SP_MATRIX_UR_H

#include "misc.h"
#include <stdbool.h>

// #define NDEBUG

/*******************************/
/*     Define & Types          */
/*******************************/

// Defintion of SparseMatrix
typedef struct SparseMatrix {
  size_t rows;
  size_t cols;
  size_t nnz;          // Number of non-zero elements
  size_t *row_indices; // Array of row pointers
  size_t *col_indices; // Array of column indices of non-zero elements
  bool is_sparse;      // True if sparse, false if dense
  double *values;      // Values
} SparseMatrix;

typedef SparseMatrix DoubleMatrix;

/*******************************/
/*      Sparse Matrix Math     */
/*******************************/

SparseMatrix *sp_create(size_t rows, size_t cols);
SparseMatrix *sp_create_rand(size_t rows, size_t cols, double density);
SparseMatrix *sp_create_from_array(size_t rows, size_t cols, size_t nnz,
                                   size_t *row_indices, size_t *col_indices,
                                   double *values);
SparseMatrix *sp_convert_to_sparse(const DoubleMatrix *dense);
bool sp_is_valid(const SparseMatrix *a);
void sp_destroy(SparseMatrix *sp_matrix);

double sp_get(const SparseMatrix *mat, size_t i, size_t j);
void sp_set(SparseMatrix *mat, size_t i, size_t j, double value);


#endif // SP_MATRIX_UR_H