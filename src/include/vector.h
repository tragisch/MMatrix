/**
 * @file vector.h
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef VECTOR_UR_H
#define VECTOR_UR_H

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "misc.h"

/*******************************/
/*     Define & Types         */
/*******************************/

#define INIT_CAPACITY 5u
#define INITIAL_SIZE 4

// Definition of DoubleVector
typedef struct {
  double *double_array;
  size_t length;
  size_t capacity;
  bool column_vec;
} DoubleVector;

/*******************************/
/*     I/O Functions           */
/*******************************/

int readInDoubleVectorData(DoubleVector *vec, const char *filepath);
int writeOutDoubleVectorData(DoubleVector *vec, const char *filepath);

/*******************************/
/*  Double Vector  (Dynamic)   */
/*******************************/

// Function Double Vector:
DoubleVector *newDoubleVector();
DoubleVector *newDoubleVectorOfLength(size_t length, double value);
DoubleVector *newRandomDoubleVectorOfLength(size_t length);
DoubleVector *cloneDoubleVector(const DoubleVector *vector);
void setDoubleVectorArray(DoubleVector *vec, double *array, size_t len_array);

// shrink, push, pop, expand
void expandDoubleVector(DoubleVector *vec);
void shrinkDoubleVector(DoubleVector *vec);
void pushValue(DoubleVector *vec, double value);
double popValue(DoubleVector *vec);

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
double multiplyDoubleVectors(DoubleVector *vec1, DoubleVector *vec2);

// free & print:
void freeDoubleVector(DoubleVector *vec);
void printDoubleVector(DoubleVector *vec);

#endif  // !VECTOR_UR_H
