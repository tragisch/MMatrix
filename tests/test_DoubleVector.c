#include "dm_math.h"
#include "dm_matrix.h"
#include "misc.h"

#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10
#define UPPER_BOUND 100

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Test preconditions:
 *******************************/

enum { INIT_CAPACITY = 2U };

void setUp(void) {
  //..
}

void tearDown(void) {}

void test_new_dm_vector() {
  DoubleVector *vec = dv_new_vector();
  TEST_ASSERT_NOT_NULL(vec);
  TEST_ASSERT_FALSE(vec->isColumnVector);
  TEST_ASSERT_EQUAL_UINT(0, vec->length);
  TEST_ASSERT_NOT_NULL(vec->mat1D);
  TEST_ASSERT_EQUAL_UINT(0, vec->mat1D->rows);
  TEST_ASSERT_EQUAL_UINT(0, vec->mat1D->columns);
  TEST_ASSERT_EQUAL_UINT(INIT_CAPACITY, vec->mat1D->rowCapacity);
  TEST_ASSERT_EQUAL_UINT(INIT_CAPACITY, vec->mat1D->columnCapacity);

  // Clean up
  dv_free_vector(vec);
}

void test_clone_dm_vector() {
  // Create a test vector
  DoubleVector *original = dv_create(3);
  original->mat1D->values[0][0] = 1.0;
  original->mat1D->values[1][0] = 2.0;
  original->mat1D->values[2][0] = 3.0;

  // Clone the vector
  DoubleVector *clone = dv_clone(original);

  // Check that the clone is equal to the original
  TEST_ASSERT_EQUAL(original->length, clone->length);
  TEST_ASSERT_EQUAL(original->isColumnVector, clone->isColumnVector);
  for (size_t i = 0; i < original->length; i++) {
    TEST_ASSERT_EQUAL(original->mat1D->values[i][0],
                      clone->mat1D->values[i][0]);
  }

  // Clean up memory
  dv_free_vector(original);
  dv_free_vector(clone);
}

void test_new_dm_vector_length() {
  DoubleVector *vec = dv_create(5);

  TEST_ASSERT_NOT_NULL(vec);
  TEST_ASSERT_FALSE(vec->isColumnVector);
  TEST_ASSERT_EQUAL_UINT32(5, vec->length);

  for (size_t i = 0; i < vec->length; i++) {
    TEST_ASSERT_EQUAL_DOUBLE(0.0, vec->mat1D->values[i][0]);
  }

  dv_free_vector(vec);
}

void test_new_rand_dm_vector_length() {
  DoubleVector *vec = dv_create_rand(5);
  TEST_ASSERT_NOT_NULL(vec);
  TEST_ASSERT_FALSE(vec->isColumnVector);
  TEST_ASSERT_EQUAL_INT(5, vec->length);

  for (size_t i = 0; i < vec->length; i++) {
    TEST_ASSERT_DOUBLE_WITHIN(1.0, 0.0, vec->mat1D->values[i][0]);
  }

  dv_free_vector(vec);
}

void test_set_dm_vector_to_array() {
  // Create a new DoubleVector
  DoubleVector *vec = dv_create(5);

  // Define an array to set the DoubleVector to
  double array[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  size_t len_array = 5;

  // Call the function
  dv_set_array(vec, array, len_array);

  // Check that the DoubleVector was set correctly
  TEST_ASSERT_EQUAL_FLOAT(array[0], vec->mat1D->values[0][0]);
  TEST_ASSERT_EQUAL_FLOAT(array[1], vec->mat1D->values[1][0]);
  TEST_ASSERT_EQUAL_FLOAT(array[2], vec->mat1D->values[2][0]);
  TEST_ASSERT_EQUAL_FLOAT(array[3], vec->mat1D->values[3][0]);
  TEST_ASSERT_EQUAL_FLOAT(array[4], vec->mat1D->values[4][0]);

  // Free the memory used by the DoubleVector
  dv_free_vector(vec);
}

// void test_pop_column() {
//   // create a matrix with 3 rows and 2 columns
//   DoubleMatrix *mat = create_dm_matrix(3, 2);
//   DoubleVector *vec1 = dv_create_from_array((double[]){1, 2, 3}, 3);
//   DoubleVector *vec2 = dv_create_from_array((double[]){4, 5, 6}, 3);

//   // pop the last column
//   DoubleVector *popped = pop_column(mat);

//   // assert that the popped vector is correct
//   TEST_ASSERT_EQUAL_DOUBLE_ARRAY((double[]){4, 5, 6},
//   popped->mat1D->values[0],
//                                  3);

//   // assert that the matrix has been updated correctly
//   TEST_ASSERT_EQUAL_INT(1, mat->columns);
//   TEST_ASSERT_EQUAL_DOUBLE_ARRAY(
//       (double[]){1, 2, 3}, dv_get_column(mat, 0)->mat1D->values[0], 3);

//   // free memory
//   dv_free_vector(popped);
//   free_dm_matrix(mat);
// }

void test_push_value(void) {
  size_t length = randomInt_upperBound(UPPER_BOUND);
  double value = 0.;
  DoubleVector *vec = dv_create(length);

  const double new_value = -1.67;
  dv_push_value(vec, new_value);
  TEST_ASSERT_EQUAL_DOUBLE(new_value, vec->mat1D->values[length][0]);

  for (size_t i = 0; i < length; i++) {
    dv_push_value(vec, randomDouble());
    TEST_ASSERT_TRUE(vec->mat1D->rowCapacity >= vec->length);
  }

  dv_free_vector(vec);
}
