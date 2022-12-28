#include "misc.h"
#include "vector.h"

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 0.0000001

#include "unity.h"
#include "unity_internals.h"

void setUp(void) {}

void tearDown(void) {}

static void test_create_vector(void) {
  DoubleVector *vec = createDoubleVector();
  TEST_ASSERT_NOT_NULL(vec);
  freeDoubleVector(vec);
}

static void test_create_vector_of_length(void) {
  size_t length = randomInt_upperBound(1000);
  size_t value = randomDouble();
  DoubleVector *vec = createDoubleVectorOfLength(length, value);
  TEST_ASSERT_NOT_NULL(vec);
  TEST_ASSERT_EQUAL_UINT32(length, vec->length);
  TEST_ASSERT_EQUAL_DOUBLE(value,
                           vec->double_array[randomInt_upperBound(length)]);
  freeDoubleVector(vec);
}

int main(void) {
  UnityBegin("matrix.c");

  RUN_TEST(test_create_vector);
  RUN_TEST(test_create_vector_of_length);

  return UnityEnd();
}
