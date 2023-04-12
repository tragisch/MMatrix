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

void test_dv_new_vector() {
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

void test_dv_clone() {
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

void test_dv_create() {
  DoubleVector *vec = dv_create(5);

  TEST_ASSERT_NOT_NULL(vec);
  TEST_ASSERT_FALSE(vec->isColumnVector);
  TEST_ASSERT_EQUAL_UINT32(5, vec->length);

  for (size_t i = 0; i < vec->length; i++) {
    TEST_ASSERT_EQUAL_DOUBLE(0.0, vec->mat1D->values[i][0]);
  }

  dv_free_vector(vec);
}

void test_dv_create_rand() {
  DoubleVector *vec = dv_create_rand(5);
  TEST_ASSERT_NOT_NULL(vec);
  TEST_ASSERT_FALSE(vec->isColumnVector);
  TEST_ASSERT_EQUAL_INT(5, vec->length);

  for (size_t i = 0; i < vec->length; i++) {
    TEST_ASSERT_DOUBLE_WITHIN(1.0, 0.0, vec->mat1D->values[i][0]);
  }

  dv_free_vector(vec);
}

void test_dv_set_array() {
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

void test_dv_pop_column() {
  // create a matrix with 3 rows and 2 columns
  double arr[2][3] = {
      {1., 2., 3.},
      {4., 5., 6.},
  };
  DoubleMatrix *mat = dm_create_from_array(2, 3, arr);

  // pop the last column
  DoubleVector *popped = dv_pop_column_matrix(mat);

  // convert popped_to_array:
  double *pop_arr = dv_get_array(popped);

  // assert that the popped vector is correct
  double *exp1 = (double[]){3, 6};
  TEST_ASSERT_EQUAL_DOUBLE_ARRAY(exp1, pop_arr, 2);

  // assert that the matrix has been updated correctly
  TEST_ASSERT_EQUAL_INT(2, mat->columns);
  double *exp2 = (double[]){1, 4};
  TEST_ASSERT_EQUAL_DOUBLE_ARRAY(exp2,
                                 dv_get_array(dv_get_column_matrix(mat, 0)), 2);

  // free memory
  dv_free_vector(popped);
  dm_free_matrix(mat);
}

void test_dv_pop_row() {
  double array[3][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  DoubleMatrix *mat = dm_create_from_array(3, 3, array);
  DoubleVector *row = dv_pop_row_matrix(mat);
  TEST_ASSERT_EQUAL_INT(mat->rows, 2); // rows should be decreased by 1
  TEST_ASSERT_EQUAL_DOUBLE(row->mat1D->values[0][0],
                           7.0); // first value in popped row should be equal to
                                 // first value in original matrix row
  TEST_ASSERT_EQUAL_DOUBLE(row->mat1D->values[1][0],
                           8.0); // second value in popped row should be equal
                                 // to second value in original matrix row
  TEST_ASSERT_EQUAL_DOUBLE(row->mat1D->values[2][0],
                           9.0); // third value in popped row should be equal to
                                 // third value in original matrix row

  // free memory:
  dv_free_vector(row);
  dm_free_matrix(mat);
}

void test_dv_push_value(void) {
  size_t length = 10;
  DoubleVector *vec = dv_create(length);

  const double new_value = -1.67;
  dv_push_value(vec, new_value);
  TEST_ASSERT_EQUAL_DOUBLE(new_value, vec->mat1D->values[length][0]);

  dv_free_vector(vec);
}

void test_dv_pop_value(void) {
  DoubleVector *vec = dv_create_rand(5);
  double expected_value = vec->mat1D->values[5][0];
  double popped_value = dv_pop_value(vec); // pop last value
  TEST_ASSERT_EQUAL_DOUBLE(popped_value, expected_value);
  TEST_ASSERT_EQUAL_INT(vec->length, 4); // length should be decreased by 1

  // free memory:
  dv_free_vector(vec);
}

void test_dv_get_array(void) {
  double values[3] = {1.0, 2.0, 3.0};
  DoubleVector *vec = dv_create_from_array(values, 3);
  double *arr = dv_get_array(vec);
  TEST_ASSERT_EQUAL_DOUBLE(
      arr[0],
      1.0); // first element in array should be equal to first element in vector
  TEST_ASSERT_EQUAL_DOUBLE(arr[1], 2.0); // second element in array should be
                                         // equal to second element in vector
  TEST_ASSERT_EQUAL_DOUBLE(
      arr[2],
      3.0); // third element in array should be equal to third element in vector

  // free memory:
  dv_free_vector(vec);
}

void test_dv_swap_elements(void) {
  double values[3] = {1.0, 2.0, 3.0};
  DoubleVector *vec = dv_create_from_array(values, 3);
  dv_swap_elements(vec, 0, 2);
  TEST_ASSERT_EQUAL_DOUBLE(
      (double)vec->mat1D->values[0][0],
      3.0); // first element should now be equal to third element
  TEST_ASSERT_EQUAL_DOUBLE(
      (double)vec->mat1D->values[2][0],
      1.0); // third element should now be equal to first element

  // free memory:
  dv_free_vector(vec);
}

void test_dv_reverse(void) {
  double values[3] = {1.0, 2.0, 3.0};
  DoubleVector *vec = dv_create_from_array(values, 3);
  dv_reverse(vec);
  TEST_ASSERT_EQUAL_DOUBLE(
      (double)vec->mat1D->values[0][0],
      3.0); // first element should now be equal to third element
  TEST_ASSERT_EQUAL_DOUBLE(
      (double)vec->mat1D->values[1][0],
      2.0); // third element should now be equal to first element
  TEST_ASSERT_EQUAL_DOUBLE(
      (double)vec->mat1D->values[2][0],
      1.0); // third element should now be equal to first element

  // free memory:
  dv_free_vector(vec);
}

void test_dv_set(void) {
  size_t len = 5;
  DoubleVector *vec = dv_create(len);

  for (size_t i = 0; i < len; i++) {
    dv_set(vec, i, (double)i);
  }

  for (size_t i = 0; i < len; i++) {
    TEST_ASSERT_EQUAL_DOUBLE((double)i, vec->mat1D->values[i][0]);
  }

  // free memory:
  dv_free_vector(vec);
}

void test_dv_get(void) {
  size_t len = 5;
  DoubleVector *vec = dv_create_rand(len);

  for (size_t i = 0; i < len; i++) {
    TEST_ASSERT_EQUAL_DOUBLE(dv_get(vec, i), vec->mat1D->values[i][0]);
  }

  // free memory:
  dv_free_vector(vec);
}
