#ifndef MATRIX_UR_H
#define MATRIX_UR_H

#include <stdbool.h>

#include "misc.h"

/*******************************/
/*     Define & Types         */
/*******************************/

// Definition of DoubleMatrix
typedef struct Matrix {
  double **values;
  size_t rows;
  size_t columns;
  size_t column_capacity;
  size_t row_capacity;
} DoubleMatrix;

// Definition of DoubleVector
typedef struct Vector {
  DoubleMatrix *mat1D;
  size_t length;
  bool column_vec;
} DoubleVector;

/*******************************/
/*        Double Matrix        */
/*******************************/

// Function DoubleMatrix:
DoubleMatrix *newDoubleMatrix();
DoubleMatrix *createDoubleMatrix(size_t rows, size_t cols);
DoubleMatrix *createRandomDoubleMatrix(size_t rows, size_t cols);
DoubleMatrix *cloneMatrix(DoubleMatrix *m);
DoubleMatrix *createIdentityMatrix(size_t rows);
DoubleMatrix *setArrayToMatrix(size_t rows, size_t cols, double **array);

// shrink, push, pop, expand
void expandMatrixRow(DoubleMatrix *mat);
void expandMatrixColumn(DoubleMatrix *mat);
void shrinkMatrixColumn(DoubleMatrix *mat);
void shrinkMatrixRow(DoubleMatrix *mat);
void pushColumn(DoubleMatrix *mat, DoubleVector *col_vec);
void pushRow(DoubleMatrix *mat, DoubleVector *row_vec);

// free
void freeDoubleMatrix(DoubleMatrix *mat);

/*******************************/
/*  Double Vector  (Dynamic)   */
/*******************************/

// Function Double Vector:
DoubleVector *newDoubleVector();
DoubleVector *newDoubleVectorOfLength(size_t length, double value);
DoubleVector *newRandomDoubleVectorOfLength(size_t length);
DoubleVector *cloneDoubleVector(const DoubleVector *vector);
DoubleVector *popColumn(DoubleMatrix *mat);
DoubleVector *popRow(DoubleMatrix *mat);
DoubleVector *getRowVector(DoubleMatrix *mat, size_t row);
DoubleVector *getColumnVector(DoubleMatrix *mat, size_t column);
void setDoubleVectorArray(DoubleVector *vec, double *array, size_t len_array);
double *getArrayFromVector(DoubleVector *vec);

// shrink, push, pop, expand
void expandDoubleVector(DoubleVector *vec);
void shrinkDoubleVector(DoubleVector *vec);
void pushValue(DoubleVector *vec, double value);
double popValue(DoubleVector *vec);

// free
void freeDoubleVector(DoubleVector *vec);

#endif  // !MATRIX_H
