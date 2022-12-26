/**
 * @file matrix.h
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 18-04-2021
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************/
/*     Define & Types         */
/*******************************/

#define INIT_CAPACITY 5u

// Definition of DoubleVector
typedef struct {
  double *double_array;
  size_t length;
  size_t capacity;
} DoubleVector;

// Definition of DoubleMatrix
typedef struct {
  double **values;
  size_t rows;
  size_t columns;
  size_t column_capacity;
  size_t row_capacity;
} DoubleMatrix;

/*******************************/
/*   Helper Functions          */
/*******************************/

// Functions Gereral:
int randomInteger(int min, int max);
double randomNumber();

// Function DoubleArray
void printDoubleArray(double *p_array, unsigned int length);
double *createRandomDoubleArray(unsigned int length);

/*******************************/
/*     I/O Functions           */
/*******************************/

int readInDoubleVectorData(DoubleVector *vec, const char *filepath);
int writeOutDoubleVectorData(DoubleVector *vec, const char *filepath);

/*******************************/
/*  Double Vector  (Dynamic)   */
/*******************************/

// Function Double Vector:
DoubleVector *createDoubleVector();
DoubleVector *createDoubleVectorOfLength(size_t length, double value);
DoubleVector *createRandomDoubleVectorOfLength(size_t length);
DoubleMatrix *cloneMatrix(DoubleMatrix *m);

// shrink, push, pop, expand
void expandDoubleVector(DoubleVector *vec);
void shrinkDoubleVector(DoubleVector *vec);
void pushValue(DoubleVector *vec, double value);
double popValue(DoubleVector *vec);

// math:
double meanOfDoubleVector(DoubleVector *vec);
double minOfDoubleVector(DoubleVector *vec);
double maxOfDoubleVector(DoubleVector *vec);
DoubleVector *addDoubleVector(DoubleVector *vec1, DoubleVector *vec2);
DoubleVector *subDoubleVector(DoubleVector *vec1, DoubleVector *vec2);
DoubleVector *multiplyScalarToVector(DoubleVector *vec, double scalar);
DoubleVector *divideScalarToVector(DoubleVector *vec, double scalar);
double multiplyDoubleVectors(DoubleVector *vec1, DoubleVector *vec2);

// free & print:
void freeDoubleVector(DoubleVector *vec);
void printDoubleVector(DoubleVector *vec);

/*******************************/
/*        Double Matrix        */
/*******************************/

// Function DoubleMatrix:
DoubleMatrix *newDoubleMatrix();
DoubleMatrix *createDoubleMatrix(size_t rows, size_t cols);
DoubleMatrix *createRandomDoubleMatrix(size_t rows, size_t cols);
DoubleMatrix *createIdentityMatrix(size_t rows);
DoubleMatrix *setArrayToMatrix(size_t rows, size_t cols,
                               double array[rows][cols]);

// shrink, push, pop, expand
static void expandMatrixRow(DoubleMatrix *mat);
static void expandMatrixColumn(DoubleMatrix *mat);
static void shrinkMatrixColumn(DoubleMatrix *mat);
static void shrinkMatrixRow(DoubleMatrix *mat);
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
