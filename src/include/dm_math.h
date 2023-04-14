
#ifndef DM_MATH_H
#define DM_MATH_H

#include "dm_matrix.h"
#include "misc.h"

/*******************************/
/*      Double Matrix Math     */
/*******************************/

double *dm_get_row_as_array(const DoubleMatrix *mat, size_t row);
void dm_multiply_by_scalar(DoubleMatrix *mat, const double scalar);
void dm_transpose(DoubleMatrix *mat);
DoubleMatrix *dm_multiply_with_matrix(const DoubleMatrix *mat1,
                                      const DoubleMatrix *mat2);
DoubleVector *dv_multiply_with_matrix(const DoubleVector *vec,
                                      const DoubleMatrix *mat);
bool dm_equal_matrix(const DoubleMatrix *m1, const DoubleMatrix *m2);

// helper:
// static double vector_multiply(const double *col, const double *row,
//                              size_t length);

/*******************************/
/*      Double Vector Math     */
/*******************************/

// math:
double dv_mean(const DoubleVector *vec);
double dv_min(const DoubleVector *vec);
double dv_max(const DoubleVector *vec);
bool dv_equal(DoubleVector *vec1, DoubleVector *vec2);
void dv_add_vector(DoubleVector *vec1, const DoubleVector *vec2);
void dv_sub_vector(DoubleVector *vec1, const DoubleVector *vec2);
void dv_multiply_by_scalar(DoubleVector *vec, const double scalar);
void dv_divide_by_scalar(DoubleVector *vec, const double scalar);
void dv_add_constant(DoubleVector *vec, const double constant);
void dv_swap_elements(DoubleVector *vec, size_t idx_i, size_t idx_j);
void dv_reverse(DoubleVector *vec);
double dv_dot_product(const DoubleVector *vec1, const DoubleVector *vec2);

#endif // DM_MATH_H