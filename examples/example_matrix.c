#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main() {
  size_t rows = 5;
  size_t cols = 5;
  DoubleMatrix *matrix = create_rand_dm_matrix(rows, cols);
  print_dm_matrix(matrix);

  // DoubleVector* col_last = newDoubleVectorOfLength(5, 2.1);
  DoubleVector *col_last = get_column_vector(matrix, matrix->columns - 1);

  DoubleVector *row_last = new_dm_vector_length(9, 8.11);

  // DoubleVector* col_last = popColumn(matrix);
  print_dm_vector(col_last);
  printf("-------\n");
  print_dm_vector(row_last);

  DoubleMatrix *mat = create_dm_matrix(5, 0);
  print_dm_matrix(mat);
  printf("row-capacity: %zu\n", col_last->mat1D->rowCapacity);

  push_value(col_last, 3.14f);
  push_value(col_last, 2.19f);
  push_value(col_last, 1.07f);
  printf("row-capacity: %zu\n", col_last->mat1D->rowCapacity);
  print_dm_vector(col_last);

  // clean up:
  free_dm_matrix(matrix);
  free_dm_matrix(mat);
  free_dm_vector(col_last);

  // initialise mytring
  char *str = "mein erster String";
  myString *mstr = mystring_init(str);

  // so anything:
  mystring_cat(mstr, " und ein weiterer Teil.");
  printf("%s\n", mstr->str);

  return 0;
}
