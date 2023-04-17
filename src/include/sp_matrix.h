#ifndef SP_MATRIX_UR_H
#define SP_MATRIX_UR_H

#include <stdbool.h>
#include "misc.h"

// #define NDEBUG

/*******************************/
/*     Define & Types         */
/*******************************/

// Defintion of SparseMatrix
typedef struct SparseMatrix {
  size_t rows;
  size_t cols;
  size_t nnz;      // Number of non-zero elements
  size_t *row_ptr; // Array of row pointers
  size_t *col_idx; // Array of column indices of non-zero elements
  bool is_sparse;  // True if sparse, false if dense
  double *values;  // Values
} SparseMatrix;

/*******************************/
/*      Sparse Matrix Math     */
/*******************************/

SparseMatrix *sp_create(size_t rows, size_t cols);
SparseMatrix *sp_create_rand(size_t rows, size_t cols, double density);
void sp_destroy(SparseMatrix *sp_matrix);

#endif // !MATRIX_H