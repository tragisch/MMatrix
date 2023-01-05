#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main() {
  // create vector with random data
  size_t length = 200;
  DoubleVector* vec = newRandomDoubleVectorOfLength(length);

  // write vec to file:
  char* file_path =
      "/Users/uwe/Documents/Programmierung/C/03_Projects/03_DoubleMatrix/data/"
      "doublevec.dat";

  writeOutDoubleVectorData(vec, file_path);

  // read vec from file:
  DoubleVector* vec2 = newDoubleVectorOfLength(6 * length, 0.);
  readInDoubleVectorData(vec2, file_path);

  printDoubleVector(vec2);

  // free:
  freeDoubleVector(vec);
  freeDoubleVector(vec2);

  return 0;
}
