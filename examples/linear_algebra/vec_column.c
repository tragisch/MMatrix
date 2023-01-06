#include <stdbool.h>
#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main(void) {
  // create array

  double a[] = {1, 2, 3};
  DoubleVector *vec = new_dm_vector();
  set_dm_vector_to_array(vec, a, -1);
  print_dm_vector(vec);

  vec->isColumnVector = true;
  print_dm_vector(vec);

  free_dm_vector(vec);

  return 0;
}
