#include <stdio.h>

#include "matrix.h"
#include "misc.h"
#include "vector.h"

int main() {
  size_t rows = 1000;
  size_t cols = 1000;
  DoubleMatrix* matrix = createRandomDoubleMatrix(rows, cols);
  printDoubleMatrix(matrix);

  DoubleVector* col_last = popColumn(matrix);
  printDoubleVector(col_last);

  // clean up:
  freeDoubleMatrix(matrix);
  freeDoubleVector(col_last);

  // initialise mytring
  char* str = "mein erster String";
  myString* mstr = mystring_init(str);

  // so anything:
  mystring_cat(mstr, " und ein weiterer Teil.");
  printf("%s\n", mstr->str);

  return 0;
}
