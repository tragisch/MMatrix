#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main() {
  size_t rows = 5;
  size_t cols = 5;
  DoubleMatrix* matrix = createRandomDoubleMatrix(rows, cols);
  printDoubleMatrix(matrix);

  // DoubleVector* col_last = newDoubleVectorOfLength(5, 2.1);
  DoubleVector* col_last = getColumnVector(matrix, matrix->columns - 1);

  DoubleVector* row_last = newDoubleVectorOfLength(9, 8.11);

  // DoubleVector* col_last = popColumn(matrix);
  printDoubleVector(col_last);
  printf("-------\n");
  printDoubleVector(row_last);

  DoubleMatrix* mat = createDoubleMatrix(5, 0);
  printDoubleMatrix(mat);
  printf("row-capacity: %zu\n", col_last->mat1D->row_capacity);

  pushValue(col_last, 3.14f);
  pushValue(col_last, 2.19f);
  pushValue(col_last, 1.07f);
  printf("row-capacity: %zu\n", col_last->mat1D->row_capacity);
  printDoubleVector(col_last);

  // clean up:
  freeDoubleMatrix(matrix);
  freeDoubleMatrix(mat);
  freeDoubleVector(col_last);

  // initialise mytring
  char* str = "mein erster String";
  myString* mstr = mystring_init(str);

  // so anything:
  mystring_cat(mstr, " und ein weiterer Teil.");
  printf("%s\n", mstr->str);

  return 0;
}
