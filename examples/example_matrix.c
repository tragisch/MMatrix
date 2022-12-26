#include <stdio.h>

#include "matrix.h"

int main() {
  size_t rows = 1000;
  size_t cols = 1000;
  DoubleMatrix* matrix = createRandomDoubleMatrix(rows, cols);

  printDoubleMatrix(matrix);

  // clean up:
  freeDoubleMatrix(matrix);
  return 0;
}
