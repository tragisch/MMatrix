#include <stdio.h>

#include "dm_io.h"
#include "dm_math.h"
#include "dm_matrix.h"
#include "misc.h"

int main() {
  // create vector with random data
  double* arr = (double*)malloc(9 * sizeof(double));
  for (size_t i = 0; i < 9; i++) {
    arr[i] = (double)randomInt_betweenBounds(0, 24);
  }
  printDoubleArray(arr, 9, 1);

  DoubleVector* vec_1 = new_dm_vector();
  set_dm_vector_to_array(vec_1, arr, 9);
  DoubleVector* vec_2 = new_dm_vector_length(9, 3.f);

  printf("------ vec1, vec2\n");
  print_dm_vector(vec_1);
  print_dm_vector(vec_2);

  printf("------ vec1  + vec2\n");
  add_dm_vector(vec_1, vec_2);
  print_dm_vector(vec_1);

  printf("------ vec1  - vec2\n");
  sub_dm_vector(vec_1, vec_2);
  print_dm_vector(vec_1);

  printf("------ min, max, mean");
  double mean_vec = mean_dm_vector(vec_1);
  double min_vec = min_dm_vector(vec_1);
  double max_vec = max_dm_vector(vec_1);
  printf("\nMean: %lf \tMin: %lf \tMax: %lf\n", mean_vec, min_vec, max_vec);

  printf("------ multiply scalar to vec_1\n");
  multiply_scalar_vector(vec_1, 3.2);
  print_dm_vector(vec_1);

  printf("------ divide scalar to vec_1\n");
  divide_scalar_vector(vec_1, 0.9);
  print_dm_vector(vec_1);

  printf("------ add_constant to vector\n");
  add_constant_vector(vec_1, -4.5);
  print_dm_vector(vec_1);

  printf("------ multiply two vectors\n");
  printf("Dot produkt: %lf\n", dot_product_dm_vectors(vec_1, vec_2));

  printf("------ reverseVector\n");
  print_dm_vector(vec_1);
  reverse_vector(vec_1);
  print_dm_vector(vec_1);

  printf("------ swap two elements\n");
  print_dm_vector(vec_1);
  swap_elements_vector(vec_1, 2, 5);
  print_dm_vector(vec_1);

  // free:
  free_dm_vector(vec_1);
  free_dm_vector(vec_2);

  return 0;
}
