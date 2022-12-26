/**
 * @file matrix.h
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef MATRIX_UR_H
#define MATRIX_UR_H

#include "pprint2D.h"
#include "utils.h"
#include "vector.h"

/*******************************/
/*     Define & Types         */
/*******************************/

#define INIT_CAPACITY 5u

// Definition of DoubleMatrix
typedef struct {
  double **values;
  size_t rows;
  size_t columns;
  size_t column_capacity;
  size_t row_capacity;
} DoubleMatrix;

/*******************************/
/*        Double Matrix        */
/*******************************/

// Function DoubleMatrix:
DoubleMatrix *newDoubleMatrix();
DoubleMatrix *createDoubleMatrix(size_t rows, size_t cols);
DoubleMatrix *createRandomDoubleMatrix(size_t rows, size_t cols);
DoubleMatrix *cloneMatrix(DoubleMatrix *m);
DoubleMatrix *createIdentityMatrix(size_t rows);
DoubleMatrix *setArrayToMatrix(size_t rows, size_t cols,
                               double array[rows][cols]);

// shrink, push, pop, expand
void expandMatrixRow(DoubleMatrix *mat);
void expandMatrixColumn(DoubleMatrix *mat);
void shrinkMatrixColumn(DoubleMatrix *mat);
void shrinkMatrixRow(DoubleMatrix *mat);
void pushColumn(DoubleMatrix *mat, DoubleVector *col_vec);
void pushRow(DoubleMatrix *mat, DoubleVector *row_vec);
DoubleVector *popColumn(DoubleMatrix *mat);
DoubleVector *popRow(DoubleMatrix *mat);

// free & print
void freeDoubleMatrix(DoubleMatrix *mat);
void printDoubleMatrix(DoubleMatrix *matrix);

// math:
DoubleVector *getRowVector(DoubleMatrix *mat, size_t row);
DoubleVector *getColumnVector(DoubleMatrix *mat, size_t column);
double *showRowVector(DoubleMatrix *mat, size_t row);
void scalarMultiply(DoubleMatrix *mat, double scalar);
void transpose(DoubleMatrix *mat);
DoubleMatrix *multiplyDoubleMatrices(DoubleMatrix *m1, DoubleMatrix *m2);
DoubleVector *multiplyVectorMatrix(DoubleVector *vec, DoubleMatrix *mat);
bool areEqualDoubleMatrices(DoubleMatrix *m1, DoubleMatrix *m2);

double vector_multiply(double *col, double *row, size_t length);

#endif  // !MATRIX_H
