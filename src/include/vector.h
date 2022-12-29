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

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "misc.h"

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

#endif  // !VECTOR_UR_H
