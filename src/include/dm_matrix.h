#ifndef MATRIX_UR_H
#define MATRIX_UR_H

#include <stdbool.h>

#include "dv_vector.h"
#include "misc.h"
#include "sp_matrix.h"

// #define NDEBUG

/*******************************/
/*     Define & Types         */
/*******************************/

// Definition of DoubleVector
typedef SparseMatrix DoubleMatrix;
typedef SparseMatrix DoubleVector;

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

#endif // !MATRIX_H
