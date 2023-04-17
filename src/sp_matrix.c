/**
 * @file sp_matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 17-04-2023
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "sp_matrix.h"

// debug:
#include <assert.h>
#include "dbg.h"

// #define NDEBUG
enum { INIT_CAPACITY = 2U };

/*******************************/
/*        Sparse Matrix        */
/*******************************/

SparseMatrix *sp_create(size_t rows, size_t cols) {
  SparseMatrix *sp_matrix = malloc(sizeof(SparseMatrix));
  sp_matrix->rows = rows;
  sp_matrix->cols = cols;
  sp_matrix->nnz = 0;
  sp_matrix->row_ptr = malloc(sizeof(size_t) * (rows + 1));
  sp_matrix->col_idx = NULL;
  sp_matrix->is_sparse = true;
  sp_matrix->values = NULL;
  return sp_matrix;
}

SparseMatrix *sp_create_rand(size_t rows, size_t cols, double density) {
  SparseMatrix *sp_matrix = sp_create(rows, cols);
  size_t max_nnz = (size_t)(rows * cols * density);
  sp_matrix->col_idx = malloc(sizeof(size_t) * max_nnz);
  sp_matrix->values = malloc(sizeof(double) * max_nnz);
  size_t nnz = 0;
  for (size_t i = 0; i < rows; i++) {
    sp_matrix->row_ptr[i] = nnz;
    for (size_t j = 0; j < cols; j++) {
      if (randomDouble() / RAND_MAX < density) {
        sp_matrix->col_idx[nnz] = j;
        sp_matrix->values[nnz] = randomDouble();
        nnz++;
      }
    }
  }
  sp_matrix->row_ptr[rows] = nnz;
  sp_matrix->nnz = nnz;
  return sp_matrix;
}

// free sparse matrix
void sp_destroy(SparseMatrix *sp_matrix) {
  free(sp_matrix->row_ptr);
  free(sp_matrix->col_idx);
  free(sp_matrix->values);
  free(sp_matrix);
}