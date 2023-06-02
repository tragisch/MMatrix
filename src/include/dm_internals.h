#ifndef MATRIX_UR_H
#define MATRIX_UR_H

#include "dm.h"

/*******************************/
/*  internals (static functions) */
/*******************************/

static DoubleMatrix *dm_create_coo(size_t rows, size_t cols);
static DoubleMatrix *dm_create_csc(size_t rows, size_t cols);
static DoubleMatrix *dm_create_dense(size_t rows, size_t cols);

// static getter / setter
static double dm_get_dense(const DoubleMatrix *mat, size_t i, size_t j);
static double dm_get_coo(const DoubleMatrix *mat, size_t i, size_t j);
static double dm_get_csc(const DoubleMatrix *matrix, size_t i, size_t j);

static void dm_set_coo(DoubleMatrix *mat, size_t i, size_t j, double value);
void insert_element(DoubleMatrix *matrix, size_t i, size_t j, double value,
                    size_t position);
static void dm_set_csc(DoubleMatrix *mat, size_t i, size_t j, double value);

static void dm_set_dense(DoubleMatrix *mat, size_t i, size_t j, double value);

// shrink, push, pop, expand
void dm_realloc_coo(DoubleMatrix *mat, size_t new_capacity);

// gauss ellimination
static void dm_gauss_elimination(DoubleMatrix *mat);

#endif // !MATRIX_H
