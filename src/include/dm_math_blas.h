#ifndef DM_MATH_BLAS_H
#define DM_MATH_BLAS_H

#include "cblas.h"
#include "dm.h"
#include <stdlib.h>

DoubleMatrix *dm_blas_multiply_by_matrix(const DoubleMatrix *mat1,
                                         const DoubleMatrix *mat2);

#endif // DM_MATH_BLAS_H