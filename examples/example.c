#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main() {
  // Create a DoubleMatrix to push a column to
  DoubleMatrix *mat = create_dm_matrix(3, 2);
  mat->values[0][0] = 1.0;
  mat->values[1][0] = 2.0;
  mat->values[2][0] = 3.0;

  print_dm_matrix(mat);

  // Create a DoubleVector to push as a column
  DoubleVector *vec1 = dv_create_from_array((double[]){1, 2, 3}, 3);

  print_dm_vector(vec1);

  // Clean up
  free_dm_matrix(mat);
  free_dm_vector(vec1);

  return 0;
}
