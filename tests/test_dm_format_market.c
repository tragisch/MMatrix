#include "dm.h"
#include "dm_convert.h"
#include "dm_format.h"
#include "dm_internals.h"
#include "dm_io.h"
#include "dm_math.h"

#include <stddef.h>
#include <stdlib.h>

/******************************
 ** Test preconditions:
 *******************************/

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10

/* Support for Meta Test Rig */
#define TEST_CASE(...)

#define EPSILON 1e-10

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Test import of Matrix Market format
 *******************************/

void test_dm_read_matrix_market(void) {
  set_default_matrix_format(COO);
  // Create a sample matrix market file for testing
  const char *filename = "test_matrix.mtx";
  FILE *file = fopen(filename, "w");
  fprintf(file, "%%MatrixMarket matrix coordinate real general\n");
  fprintf(file, "3 3 5\n");
  fprintf(file, "1 1 1.0\n");
  fprintf(file, "1 2 2.0\n");
  fprintf(file, "2 1 3.0\n");
  fprintf(file, "2 2 4.0\n");
  fprintf(file, "3 3 5.0\n");
  fclose(file);

  // Read the matrix from the matrix market file
  DoubleMatrix *mat = dm_read_matrix_market(filename);

  // Create the expected COO matrix
  DoubleMatrix *expected = dm_create(3, 3);
  dm_set(expected, 0, 0, 1.0);
  dm_set(expected, 0, 1, 2.0);
  dm_set(expected, 1, 0, 3.0);
  dm_set(expected, 1, 1, 4.0);
  dm_set(expected, 2, 2, 5.0);

  // Compare the actual and expected COO matrices
  TEST_ASSERT_TRUE(dm_equal(mat, expected) == true);

  // Clean up
  dm_destroy(mat);
  dm_destroy(expected);
  remove(filename);
}
