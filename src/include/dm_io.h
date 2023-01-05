#ifndef DM_IO_H
#define DM_IO_H

#include "dm_matrix.h"
#include "misc.h"

/*******************************/
/*     I/O Functions           */
/*******************************/

// read & write
int readInDoubleVectorData(DoubleVector *vec, const char *filepath);
int writeOutDoubleVectorData(DoubleVector *vec, const char *filepath);

// print stdout:
void printDoubleMatrix(DoubleMatrix *matrix);
void printDoubleVector(DoubleVector *vec);

#endif  // DM_IO_H
