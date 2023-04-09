#include <stdbool.h>
#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main(void) {
  // create array

  double a[] = {1, 2, 3};
  DoubleVector *vec = dv_new_vector();
  dv_set_array(vec, a, -1);
  print_dm_vector(vec);

  vec->isColumnVector = true;
  print_dm_vector(vec);

  dv_free_vector(vec);

  return 0;
}
