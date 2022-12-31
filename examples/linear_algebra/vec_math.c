#include <stdio.h>

#include "misc.h"
#include "vector.h"

int main() {
  // create vector with random data
  double* arr = (double*)malloc(9 * sizeof(double));
  for (size_t i = 0; i < 9; i++) {
    arr[i] = (double)randomInt_betweenBounds(0, 24);
  }
  printDoubleArray(arr, 9, 1);

  DoubleVector* vec_1 = newDoubleVector();
  setDoubleVectorArray(vec_1, arr, 9);
  DoubleVector* vec_2 = newDoubleVectorOfLength(9, 3.);

  printf("------ vec1, vec2\n");
  printDoubleVector(vec_1);
  printDoubleVector(vec_2);

  printf("------ vec1  + vec2\n");
  addDoubleVector(vec_1, vec_2);
  printDoubleVector(vec_1);

  printf("------ vec1  - vec2\n");
  subDoubleVector(vec_1, vec_2);
  printDoubleVector(vec_1);

  printf("------ min, max, mean");
  double mean_vec = meanOfDoubleVector(vec_1);
  double min_vec = minOfDoubleVector(vec_1);
  double max_vec = maxOfDoubleVector(vec_1);
  printf("\nMean: %lf \tMin: %lf \tMax: %lf\n", mean_vec, min_vec, max_vec);

  printf("------ multiply scalar to vec_1\n");
  multiplyScalarToVector(vec_1, 3.2);
  printDoubleVector(vec_1);

  printf("------ divide scalar to vec_1\n");
  divideScalarToVector(vec_1, 0.9);
  printDoubleVector(vec_1);

  printf("------ add_constant to vector\n");
  addConstantToVector(vec_1, -4.5);
  printDoubleVector(vec_1);

  printf("------ multiply two vectors\n");
  printf("Dot produkt: %lf\n", multiplyDoubleVectors(vec_1, vec_2));

  printf("------ reverseVector\n");
  printDoubleVector(vec_1);
  reverseVector(vec_1);
  printDoubleVector(vec_1);

  printf("------ swap two elements\n");
  printDoubleVector(vec_1);
  swapElementsOfVector(vec_1, 2, 5);
  printDoubleVector(vec_1);

  // free:
  freeDoubleVector(vec_1);
  freeDoubleVector(vec_2);

  return 0;
}
