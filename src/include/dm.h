#ifndef DM_H
#define DM_H

#include "khash.h"
#include <stdbool.h>
#include <stdlib.h>

/*******************************/
/*     Define & Types          */
/*******************************/

// sparse matrix formats
typedef enum { DENSE, SPARSE, HASHTABLE, VECTOR } matrix_format;
KHASH_MAP_INIT_INT64(entry, double)

// Standard is of SparseMatrix (COO-Format)
typedef struct DoubleMatrix {
  size_t rows;
  size_t cols;
  size_t capacity;             // Capacity of row_indices and col_indices
  size_t nnz;                  // Number of non-zero elements
  size_t *row_indices;         // Array of row indices of non-zero elements
  size_t *col_indices;         // Array of column indices of non-zero elements
  matrix_format format;        // SPARSE or DENSE or HASHTABLE or VECTOR
  khash_t(entry) * hash_table; // Hash table for fast access
  double *values;              // Values
} DoubleMatrix;

// Definition of DoubleVector
typedef DoubleMatrix DoubleVector;

/*******************************/
/*     Create  & convert       */
/*******************************/

// Create, Clone, Destroy
DoubleMatrix *dm_create(size_t rows, size_t cols); // empty sparse matrix
DoubleMatrix *dm_create_nnz(size_t rows, size_t cols, size_t nnz);
DoubleMatrix *dm_create_format(size_t rows, size_t cols, matrix_format format);
DoubleMatrix *dm_clone(DoubleMatrix *m);

// free memory
void dm_destroy(DoubleMatrix *mat);

/*******************************/
/*        Getter & Setter      */
/*******************************/

double dm_get(const DoubleMatrix *mat, size_t i, size_t j);
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value);

/*******************************/
/*           Transform         */
/*******************************/

void dm_resize(DoubleMatrix *mat, size_t new_row, size_t new_col);
void dm_convert(DoubleMatrix *mat, matrix_format format);

// Push Vectors
void dm_push_column(DoubleMatrix *mat, DoubleVector *col_vec);
void dm_push_row(DoubleMatrix *mat, DoubleVector *row_vec);

/*******************************/
/*     Special Matrices        */
/*******************************/

DoubleMatrix *dm_create_rand(size_t rows, size_t cols, double density);
DoubleMatrix *dm_create_identity(size_t rows);
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols,
                                   double array[rows][cols]);
DoubleMatrix *dm_get_sub_matrix(DoubleMatrix *mat, size_t row_start,
                                size_t row_end, size_t col_start,
                                size_t col_end);
DoubleMatrix *dm_create_diagonal(size_t rows, size_t cols, double array[rows]);

#endif // DM_H
