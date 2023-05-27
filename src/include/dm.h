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

extern matrix_format default_matrix_format;

// struct of DoubleMatrix
typedef struct DoubleMatrix {
  size_t rows;                 // Number of rows
  size_t cols;                 // Number of columns
  size_t capacity;             // Capacity of row_indices and col_indices
  size_t nnz;                  // Number of non-zero elements
  size_t *row_indices;         // Array of row indices of non-zero elements
  size_t *col_indices;         // Array of column indices of non-zero elements
  matrix_format format;        // SPARSE or DENSE or HASHTABLE or VECTOR
  khash_t(entry) * hash_table; // Hash table for fast access
  double *values;              // Values
} DoubleMatrix;

// Definition of DoubleVector
typedef struct DoubleMatrix HashTableMatrix;

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

/*******************************/
/*     Special Matrices        */
/*******************************/

DoubleMatrix *dm_create_rand(size_t rows, size_t cols, double density);
DoubleMatrix *dm_create_rand_between(size_t rows, size_t cols, size_t min,
                                     size_t max);
DoubleMatrix *dm_create_identity(size_t rows);
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols,
                                   double array[rows][cols]);

DoubleMatrix *dm_create_diagonal(size_t rows, size_t cols, double array[rows]);

#endif // DM_H
