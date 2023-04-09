#include "misc.h"

#include "unity.h"
#include "unity_internals.h"

void test_randomDouble(void) {
  const int num_runs = 1000;
  double min_value = 0.0;
  double max_value = 1.0;
  double random_num;

  for (int i = 0; i < num_runs; i++) {
    random_num = randomDouble();
    TEST_ASSERT_TRUE(random_num >= min_value && random_num <= max_value);
  }
}
