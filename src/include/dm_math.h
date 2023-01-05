
#ifndef DM_MATH_H
#define DM_MATH_H

#include "dm_matrix.h"
#include "misc.h"

/*******************************/
/*      Double Matrix Math     */
/*******************************/

DoubleVector *getRowVector(DoubleMatrix *mat, size_t row);
DoubleVector *getColumnVector(DoubleMatrix *mat, size_t column);
double *showRowVector(DoubleMatrix *mat, size_t row);
void scalarMultiply(DoubleMatrix *mat, double scalar);
void transpose(DoubleMatrix *mat);
DoubleMatrix *multiplyDoubleMatrices(DoubleMatrix *m1, DoubleMatrix *m2);
DoubleVector *multiplyVectorMatrix(DoubleVector *vec, DoubleMatrix *mat);
bool areEqualDoubleMatrices(DoubleMatrix *m1, DoubleMatrix *m2);

double vector_multiply(double *col, double *row, size_t length);

/*******************************/
/*      Double Vector Math     */
/*******************************/

// math:
double meanOfDoubleVector(DoubleVector *vec);
double minOfDoubleVector(DoubleVector *vec);
double maxOfDoubleVector(DoubleVector *vec);
void addDoubleVector(DoubleVector *vec1, const DoubleVector *vec2);
void subDoubleVector(DoubleVector *vec1, const DoubleVector *vec2);
void multiplyScalarToVector(DoubleVector *vec, const double scalar);
void divideScalarToVector(DoubleVector *vec, const double scalar);
void addConstantToVector(DoubleVector *vec, const double constant);
void swapElementsOfVector(DoubleVector *vec, size_t i, size_t j);
void reverseVector(DoubleVector *vec);
double dotproductDoubleVectors(DoubleVector *vec1, DoubleVector *vec2);





#endif  // DM_MATH_H