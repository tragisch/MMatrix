#include <stdbool.h>
#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main(void) {
  // create array

  double a[] = {1, 2, 3};
  DoubleVector *vec = newDoubleVector();
  setDoubleVectorArray(vec, a, 0);
  printDoubleVector(vec);

  vec->column_vec = true;
  printDoubleVector(vec);

  freeDoubleVector(vec);

  return 0;
}
