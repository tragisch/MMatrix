#include "dm_math.h"
#include "dm_matrix.h"
#include "unity.h"

void setUp(void) {}

void tearDown(void) {}

static void test_matric_create(void) {
  DoubleMatrix *mat = new_dm_matrix();
  TEST_ASSERT_NOT_NULL(mat);
}

int main(void) {
  UnityBegin("matrix.c");

  RUN_TEST(test_matric_create);

  return UnityEnd();
}
