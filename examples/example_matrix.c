#include <stdio.h>

#include "matrix.h"
#include "pprint2Dmatrix.h"

int main() {
  size_t rows = 1000;
  size_t cols = 1000;
  DoubleMatrix* matrix = createRandomDoubleMatrix(rows, cols);

  printDoubleMatrix(matrix);

  // pprint2D(matrix->values, rows, cols);

  // clean up:
  freeDoubleMatrix(matrix);
  return 0;
}
