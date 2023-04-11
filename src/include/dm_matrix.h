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
  double **values;
  size_t rows;
  size_t columns;
  size_t columnCapacity;
  size_t rowCapacity;
} DoubleMatrix;

// Definition of DoubleVector
typedef struct Vector {
  DoubleMatrix *mat1D;
  size_t length;
  bool isColumnVector;
} DoubleVector;

/*******************************/
/*        Double Matrix        */
/*******************************/

// Function DoubleMatrix:
DoubleMatrix *dm_matrix();
DoubleMatrix *dm_create(size_t rows, size_t cols);
DoubleMatrix *dm_create_rand(size_t rows, size_t cols);
DoubleMatrix *dm_clone(DoubleMatrix *m);
DoubleMatrix *dm_create_identity(size_t rows);
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols, double **array);

// shrink, push, pop, expand
static void expand_dm_matrix_row(DoubleMatrix *mat);
static void expand_dm_matrix_column(DoubleMatrix *mat);
static void shrink_dm_matrix_column(DoubleMatrix *mat);
static void shrink_dm_matrix_row(DoubleMatrix *mat);
void dm_push_column(DoubleMatrix *mat, const DoubleVector *col_vec);
void dm_push_row(DoubleMatrix *mat, const DoubleVector *row_vec);
double dm_get(DoubleMatrix *mat, size_t i, size_t j);
void dm_set(DoubleMatrix *mat, size_t i, size_t j, const double value);

// free
void dm_free_matrix(DoubleMatrix *mat);

/*******************************/
/*  Double Vector  (Dynamic)   */
/*******************************/

// Function Double Vector:
DoubleVector *dv_new_vector();
DoubleVector *dv_create(size_t length);
DoubleVector *dv_create_rand(size_t length);
DoubleVector *dv_create_from_array(const double *array, const size_t length);
DoubleVector *dv_clone(const DoubleVector *vector);
DoubleVector *dv_pop_column(DoubleMatrix *mat);
DoubleVector *dv_pop_row(DoubleMatrix *mat);
DoubleVector *dv_get_row(const DoubleMatrix *mat, size_t row);
DoubleVector *dv_get_column(const DoubleMatrix *mat, size_t column);
void dv_set_array(DoubleVector *vec, const double *array, size_t len_array);
double *dv_get_array(const DoubleVector *vec);

// shrink, push, pop, expand
static void expand_dm_vector(DoubleVector *vec);
static void shrink_dm_vector(DoubleVector *vec);
void dv_push_value(DoubleVector *vec, double value);
double dv_pop_value(DoubleVector *vec);

// free
void dv_free_vector(DoubleVector *vec);

#endif // !MATRIX_H
