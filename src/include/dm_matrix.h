#ifndef MATRIX_UR_H
#define MATRIX_UR_H

#include <stdbool.h>

#include "misc.h"

// #define NDEBUG

/*******************************/
/*     Define & Types         */
/*******************************/

// Definition of DoubleMatrix
typedef struct Matrix {
  double *values;
  size_t rows;
  size_t cols;
} DoubleMatrix;

// Definition of DoubleVector
typedef DoubleMatrix DoubleVector;

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

// Test if vector or matrix (true = vector)
bool dm_is_vector(DoubleMatrix *mat);

// shrink, push, pop, expand
void dm_resize(DoubleMatrix *mat, size_t rows, size_t cols);

// Getters and Setters
void dm_push_column(DoubleMatrix *mat, DoubleVector *col_vec);
void dm_push_row(DoubleMatrix *mat, DoubleVector *row_vec);
DoubleVector *dm_pop_column_matrix(DoubleMatrix *mat);
DoubleVector *dm_pop_row_matrix(DoubleMatrix *mat);
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
bool dv_is_row_vector(DoubleMatrix *vector);

// Get DoubleVector from DoubleMatrix:
DoubleVector *dv_get_row_matrix(DoubleMatrix *mat, size_t row);
DoubleVector *dv_get_column_matrix(DoubleMatrix *mat, size_t column);

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

#endif // !MATRIX_H
