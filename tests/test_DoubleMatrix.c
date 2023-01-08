#include "dm_math.h"
#include "dm_matrix.h"
#include "unity.h"
#include "unity_internals.h"

static void test_matrix_create(void) {
  DoubleMatrix *mat = new_dm_matrix();
  TEST_ASSERT_NOT_NULL(mat);
  free_dm_matrix(mat);
}
