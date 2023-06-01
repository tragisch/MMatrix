#include "dm.h"
#include "dm_convert.h"
#include "dm_internals.h"
#include "dm_io.h"
#include "dm_math.h"
#include "dm_vector.h"
#include <stddef.h>
#include <stdlib.h>

/******************************
 ** Test preconditions:
 *******************************/

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10

/* Support for Meta Test Rig */
#define TEST_CASE(...)

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** realloc_coo
 *******************************/

void test_dm_realloc_coo(void) {
  // Create a matrix with initial capacity 3
  DoubleMatrix *mat = dm_create(3, 3);

  // Reallocate the COO matrix with a new capacity of 5
  dm_realloc_coo(mat, 5000);

  // Check if the COO matrix has been reallocated correctly
  TEST_ASSERT_EQUAL(6000, mat->capacity);

  // Clean up
  dm_destroy(mat);
}
