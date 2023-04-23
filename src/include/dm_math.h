
#ifndef DM_MATH_H
#define DM_MATH_H

#include "dm_matrix.h"
#include "misc.h"

/*******************************/
/*      Double Matrix Math     */
/*******************************/

DoubleMatrix *dm_multiply_with_matrix(const DoubleMatrix *mat1,
                                      const DoubleMatrix *mat2);
DoubleVector *dv_multiply_with_matrix(const DoubleVector *vec,
                                      const DoubleMatrix *mat);
void dm_multiply_by_scalar(DoubleMatrix *mat, const double scalar);
bool dm_equal_matrix(const DoubleMatrix *m1, const DoubleMatrix *m2);
void dm_transpose(DoubleMatrix *mat);
double dm_determinant(const DoubleMatrix *mat);
double dm_density(const DoubleMatrix *mat);
double dm_trace(const DoubleMatrix *mat);

// principles
DoubleMatrix *dm_inverse(DoubleMatrix *mat);
int dm_rank(DoubleMatrix *mat);

#endif // DM_MATH_H