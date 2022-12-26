#include <stdio.h>
#include "matrix.h"



int main() {
  size_t rows = 10;
  size_t cols = 10;
  DoubleMatrix* matrix = createRandomDoubleMatrix(rows, cols);

  printDoubleMatrix(matrix);
  return 0;
}
