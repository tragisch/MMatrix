
#ifndef DM_MATH_H
#define DM_MATH_H

#include "dm_matrix.h"
#include "misc.h"

/*******************************/
/*      Double Matrix Math     */
/*******************************/

DoubleVector *dv_get_row(const DoubleMatrix *mat, size_t row);
DoubleVector *dv_get_column(const DoubleMatrix *mat, size_t column);
double *get_row_array(const DoubleMatrix *mat, size_t row);
void multiply_scalar_matrix(DoubleMatrix *mat, double scalar);
void transpose(DoubleMatrix *mat);
DoubleMatrix *multiply_dm_matrices(const DoubleMatrix *mat1,
                                   const DoubleMatrix *mat2);
DoubleVector *multiply_dm_vector_matrix(const DoubleVector *vec,
                                        const DoubleMatrix *mat);
bool are_dm_matrix_equal(const DoubleMatrix *m1, const DoubleMatrix *m2);

// helper:
static double vector_multiply(const double *col, const double *row,
                              size_t length);

/*******************************/
/*      Double Vector Math     */
/*******************************/

// math:
double mean_dm_vector(const DoubleVector *vec);
double min_dm_vector(const DoubleVector *vec);
double max_dm_vector(const DoubleVector *vec);
void add_dm_vector(DoubleVector *vec1, const DoubleVector *vec2);
void sub_dm_vector(DoubleVector *vec1, const DoubleVector *vec2);
void multiply_scalar_vector(DoubleVector *vec, const double scalar);
void divide_scalar_vector(DoubleVector *vec, const double scalar);
void add_constant_vector(DoubleVector *vec, const double constant);
void swap_elements_vector(DoubleVector *vec, size_t idx_i, size_t idx_j);
void reverse_vector(DoubleVector *vec);
double dot_product_dm_vectors(const DoubleVector *vec1,
                              const DoubleVector *vec2);

#endif // DM_MATH_H