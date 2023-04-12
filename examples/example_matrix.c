#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main() {
  size_t rows = 5;
  size_t cols = 5;
  DoubleMatrix *matrix = dm_create_rand(rows, cols);
  print_dm_matrix(matrix);

  // DoubleVector* col_last = newDoubleVectorOfLength(5, 2.1);
  DoubleVector *col_last = dv_get_column_matrix(matrix, matrix->columns - 1);

  DoubleVector *row_last = dv_create(9);

  // DoubleVector* col_last = popColumn(matrix);
  print_dm_vector(col_last);
  printf("-------\n");
  print_dm_vector(row_last);

  DoubleMatrix *mat = dm_create(5, 0);
  print_dm_matrix(mat);
  printf("row-capacity: %zu\n", col_last->mat1D->rowCapacity);

  dv_push_value(col_last, 3.14f);
  dv_push_value(col_last, 2.19f);
  dv_push_value(col_last, 1.07f);
  printf("row-capacity: %zu\n", col_last->mat1D->rowCapacity);
  print_dm_vector(col_last);

  // clean up:
  dm_free_matrix(matrix);
  dm_free_matrix(mat);
  dv_free_vector(col_last);

  // initialise mytring
  char *str = "mein erster String";
  myString *mstr = mystring_init(str);

  // so anything:
  mystring_cat(mstr, " und ein weiterer Teil.");
  printf("%s\n", mstr->str);

  return 0;
}
