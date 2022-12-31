#include "misc.h"
#include "vector.h"

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 0.0000001

#include "unity.h"
#include "unity_internals.h"

void setUp(void) {}

void tearDown(void) {}

static void test_create_vector(void) {
  DoubleVector *vec = newDoubleVector();
  TEST_ASSERT_NOT_NULL(vec);
  freeDoubleVector(vec);
}

static void test_create_vector_of_length(void) {
  size_t length = randomInt_upperBound(1000);
  double value = randomDouble();
  DoubleVector *vec = newDoubleVectorOfLength(length, value);
  TEST_ASSERT_NOT_NULL(vec);
  TEST_ASSERT_EQUAL_UINT32(length, vec->length);
  TEST_ASSERT_EQUAL_DOUBLE(value,
                           vec->double_array[randomInt_upperBound(length)]);
  freeDoubleVector(vec);
}

static void test_push_value(void) {
  size_t length = randomInt_upperBound(100);
  double value = 0.;
  DoubleVector *vec = newDoubleVectorOfLength(length, value);

  double new_value = -1.67;
  pushValue(vec, new_value);
  TEST_ASSERT_EQUAL_DOUBLE(new_value, vec->double_array[length]);

  for (size_t i = 0; i < length; i++) {
    pushValue(vec, randomDouble());
    TEST_ASSERT_TRUE(vec->capacity >= vec->length);
  }

  freeDoubleVector(vec);
}

int main(void) {
  UnityBegin("matrix.c");

  RUN_TEST(test_create_vector);
  RUN_TEST(test_create_vector_of_length);
  RUN_TEST(test_push_value);

  return UnityEnd();
}
