#include <stdbool.h>
#include <stdio.h>

#include "misc.h"
#include "vector.h"

int main(void) {
  // create array

  double a[] = {1, 2, 3};
  DoubleVector *vec = newDoubleVector();
  setDoubleVectorArray(vec, a, 3);
  printDoubleVector(vec);

  vec->column_vec = true;
  printDoubleVector(vec);

  freeDoubleVector(vec);

  return 0;
}
