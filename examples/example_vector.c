#include <stdio.h>

#include "misc.h"
#include "vector.h"

int main() {
  // create vector with random data
  size_t length = 200;
  DoubleVector* vec = createRandomDoubleVectorOfLength(length);

  // write vec to file:
  char* file_path =
      "/Users/uwe/Documents/Programmierung/C/03_Projects/03_DoubleMatrix/data/"
      "doublevec.dat";

  writeOutDoubleVectorData(vec, file_path);

  // read vec from file:
  DoubleVector* vec2 = createDoubleVectorOfLength(6 * length, 0.);
  readInDoubleVectorData(vec2, file_path);

  printDoubleArray(vec2->double_array, length, 1);

  // free:
  freeDoubleVector(vec);
  freeDoubleVector(vec2);

  return 0;
}
