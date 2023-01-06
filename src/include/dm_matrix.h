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
DoubleMatrix *new_dm_matrix();
DoubleMatrix *create_dm_matrix(size_t rows, size_t cols);
DoubleMatrix *create_rand_dm_matrix(size_t rows, size_t cols);
DoubleMatrix *clone_dm_matrix(DoubleMatrix *m);
DoubleMatrix *create_identity_matrix(size_t rows);
DoubleMatrix *set_array_to_dm_matrix(size_t rows, size_t cols, double **array);

// shrink, push, pop, expand
static void expand_dm_matrix_row(DoubleMatrix *mat);
static void expand_dm_matrix_column(DoubleMatrix *mat);
static void shrink_dm_matrix_column(DoubleMatrix *mat);
static void shrink_dm_matrix_row(DoubleMatrix *mat);
void push_column(DoubleMatrix *mat, const DoubleVector *col_vec);
void push_row(DoubleMatrix *mat, const DoubleVector *row_vec);

// free
void free_dm_matrix(DoubleMatrix *mat);

/*******************************/
/*  Double Vector  (Dynamic)   */
/*******************************/

// Function Double Vector:
DoubleVector *new_dm_vector();
DoubleVector *new_dm_vector_length(size_t length, double value);
DoubleVector *new_rand_dm_vector_length(size_t length);
DoubleVector *clone_dm_vector(const DoubleVector *vector);
DoubleVector *pop_column(DoubleMatrix *mat);
DoubleVector *pop_row(DoubleMatrix *mat);
DoubleVector *get_row_vector(const DoubleMatrix *mat, size_t row);
DoubleVector *get_column_vector(const DoubleMatrix *mat, size_t column);
void set_dm_vector_to_array(DoubleVector *vec, const double *array,
                            size_t len_array);
double *get_array_from_vector(const DoubleVector *vec);

// shrink, push, pop, expand
static void expand_dm_vector(DoubleVector *vec);
static void shrink_dm_vector(DoubleVector *vec);
void push_value(DoubleVector *vec, double value);
double pop_value(DoubleVector *vec);

// free
void free_dm_vector(DoubleVector *vec);

#endif  // !MATRIX_H
