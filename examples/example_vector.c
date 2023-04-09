#include <stdio.h>

#include "dm_io.h"
#include "dm_matrix.h"
#include "misc.h"

int main() {
  // create vector with random data
  size_t length = 200;
  DoubleVector *vec = dv_create_rand(length);

  // write vec to file:
  char *file_path =
      "/Users/uwe/Documents/Programmierung/C/03_Projects/03_DoubleMatrix/data/"
      "doublevec.dat";

  write_dm_vector_to_file(vec, file_path);

  // read vec from file:
  DoubleVector *vec2 = dv_create(6 * length);
  read_dm_vector_from_file(vec2, file_path);

  print_dm_vector(vec2);

  // free:
  dv_free_vector(vec);
  dv_free_vector(vec2);

  return 0;
}
