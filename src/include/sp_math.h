#ifndef SP_MATH_UR_H
#define SP_MATH_UR_H

#include "misc.h"
#include "sp_matrix.h"

/*******************************/
/*     Define & Types         */
/*******************************/

// Definition of DoubleVector
typedef SparseMatrix DoubleMatrix;
typedef SparseMatrix DoubleVector;

/*******************************/
/*      Sparse Matrix Math     */
/*******************************/

double sp_density(const SparseMatrix *mat);

#endif // SP_MATH_UR_H
