
#ifndef DM_MATH_H
#define DM_MATH_H

#include "dm_matrix.h"
#include "misc.h"

/*******************************/
/*      General stuff     */
/*******************************/

bool dm_is_zero(double value);

/*******************************/
/*      Double Matrix Math     */
/*******************************/

DoubleMatrix *dm_multiply_by_matrix(const DoubleMatrix *mat1,
                                    const DoubleMatrix *mat2);
DoubleVector *dm_multiply_by_vector(const DoubleMatrix *mat,
                                    const DoubleVector *vec);
void dm_multiply_by_scalar(DoubleMatrix *mat, const double scalar);
bool dm_equal_matrix(const DoubleMatrix *m1, const DoubleMatrix *m2);
void dm_transpose(DoubleMatrix *mat);
double dm_determinant(const DoubleMatrix *mat);
double dm_density(const DoubleMatrix *mat);
double dm_trace(const DoubleMatrix *mat);

// principles
DoubleMatrix *dm_inverse(DoubleMatrix *mat);
int dm_rank(const DoubleMatrix *mat);

/*******************************/
/*      Sparse Matrix Math     */
/*******************************/

double sp_density(const DoubleMatrix *mat);

/*******************************/
/*      Double Vector Math     */
/*******************************/

// Test equality:
bool dv_equal(DoubleVector *vec1, DoubleVector *vec2);

// math:
double dv_mean(const DoubleVector *vec);
double dv_min(const DoubleVector *vec);
double dv_max(const DoubleVector *vec);

void dv_add_vector(DoubleVector *vec1, const DoubleVector *vec2);
void dv_sub_vector(DoubleVector *vec1, const DoubleVector *vec2);
void dv_multiply_by_scalar(DoubleVector *vec, const double scalar);
void dv_divide_by_scalar(DoubleVector *vec, const double scalar);
void dv_add_constant(DoubleVector *vec, const double constant);
void dv_swap_elements(DoubleVector *vec, size_t idx_i, size_t idx_j);
void dv_reverse(DoubleVector *vec);
void dv_transpose(DoubleVector *vec);
double dv_dot_product(const DoubleVector *vec1, const DoubleVector *vec2);
double dv_magnitude(DoubleVector *vec);
void dv_normalize(DoubleVector *vec);

#endif // DM_MATH_H
