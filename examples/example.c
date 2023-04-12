#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main() {

  double arr[3][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  DoubleMatrix *mat2 = dm_create_from_array(3, 3, arr);
  print_dm_matrix(mat2);
  DoubleVector *row1 = dv_get_row_matrix(mat2, 2);
  print_dm_vector(row1);

  DoubleVector *row = dv_pop_row_matrix(mat2);
  print_dm_matrix(mat2);
  print_dm_vector(row);
  // Clean up
  dm_free_matrix(mat2);

  return 0;
}
