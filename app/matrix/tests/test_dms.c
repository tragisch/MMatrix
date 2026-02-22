/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "dms.h"

/******************************
 ** Test preconditions:
 *******************************/
#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10

/* Support for Meta Test Rig */
// #define TEST_CASE(...)

#if __has_include("unity.h")
#include "unity.h"
#include "unity_internals.h"
#endif

#if defined(__has_include)
#if __has_include(<cs.h>)
#include <cs.h>
#define TEST_DMS_HAS_CSPARSE 1
#endif
#endif

#ifndef TEST_ASSERT_EQUAL
#define TEST_ASSERT_EQUAL(expected, actual) \
  do {                                      \
    (void)(expected);                       \
    (void)(actual);                         \
  } while (0)
#endif
#ifndef TEST_ASSERT_NULL
#define TEST_ASSERT_NULL(value) \
  do {                          \
    (void)(value);              \
  } while (0)
#endif
#ifndef TEST_ASSERT_NOT_NULL
#define TEST_ASSERT_NOT_NULL(value) \
  do {                              \
    (void)(value);                  \
  } while (0)
#endif
#ifndef TEST_ASSERT_TRUE
#define TEST_ASSERT_TRUE(condition) \
  do {                              \
    (void)(condition);              \
  } while (0)
#endif
#ifndef TEST_ASSERT_NOT_EQUAL
#define TEST_ASSERT_NOT_EQUAL(expected, actual) \
  do {                                          \
    (void)(expected);                           \
    (void)(actual);                             \
  } while (0)
#endif
#ifndef TEST_IGNORE_MESSAGE
#define TEST_IGNORE_MESSAGE(message) \
  do {                               \
    (void)(message);                 \
  } while (0)
#endif

/******************************
 ** Creation of matrices:
 *******************************/

void test_dms_create(void) {
  DoubleSparseMatrix *m = dms_create(3, 3, 3);
  TEST_ASSERT_EQUAL(3, m->rows);
  TEST_ASSERT_EQUAL(3, m->cols);
  TEST_ASSERT_EQUAL(0, m->nnz);
  TEST_ASSERT_EQUAL(3, m->capacity);
  dms_destroy(m);
}

void test_dms_clone(void) {
  DoubleSparseMatrix *m = dms_create(3, 3, 3);
  DoubleSparseMatrix *m2 = dms_create_clone(m);
  TEST_ASSERT_EQUAL(3, m2->rows);
  TEST_ASSERT_EQUAL(3, m2->cols);
  TEST_ASSERT_EQUAL(0, m2->nnz);
  TEST_ASSERT_EQUAL(3, m2->capacity);
  dms_destroy(m);
  dms_destroy(m2);
}

void test_dms_identity_size_3(void) {
  size_t n = 3;
  DoubleSparseMatrix *mat = dms_create_identity(n);

  // Verify the dimensions
  TEST_ASSERT_EQUAL(n, mat->rows);
  TEST_ASSERT_EQUAL(n, mat->cols);

  // Verify the number of non-zero elements
  TEST_ASSERT_EQUAL(n, mat->nnz);

  // Verify the values and positions
  for (size_t i = 0; i < n; i++) {
    TEST_ASSERT_EQUAL(i, mat->row_indices[i]);
    TEST_ASSERT_EQUAL(i, mat->col_indices[i]);
    TEST_ASSERT_EQUAL(1.0, mat->values[i]);
  }

  dms_destroy(mat);
}

// Test to check if identity matrix of size 1 is created correctly
void test_dms_identity_size_1(void) {
  size_t n = 1;
  DoubleSparseMatrix *mat = dms_create_identity(n);

  // Verify the dimensions
  TEST_ASSERT_EQUAL(n, mat->rows);
  TEST_ASSERT_EQUAL(n, mat->cols);

  // Verify the number of non-zero elements
  TEST_ASSERT_EQUAL(n, mat->nnz);

  // Verify the values and positions
  TEST_ASSERT_EQUAL(0, mat->row_indices[0]);
  TEST_ASSERT_EQUAL(0, mat->col_indices[0]);
  TEST_ASSERT_EQUAL(1.0, mat->values[0]);

  dms_destroy(mat);
}

// Test to check if identity matrix of size 0 is handled correctly
void test_dms_identity_size_0(void) {
  size_t n = 0;
  DoubleSparseMatrix *mat = dms_create_identity(n);

  // Verify the result is NULL for invalid size
  TEST_ASSERT_NULL(mat);
}

// Helper function to count the number of unique non-zero elements
size_t count_unique_non_zeros(const DoubleSparseMatrix *mat) {
  size_t count = 0;
  for (size_t i = 0; i < mat->nnz; i++) {
    if (mat->values[i] != 0.0) {
      count++;
    }
  }
  return count;
}

// Test to check if a random sparse matrix of specified density is created
// correctly
void test_dms_rand_density(void) {
  size_t rows = 10;
  size_t cols = 10;
  double density = 0.2;
  DoubleSparseMatrix *mat = dms_create_random(rows, cols, density);

  // Verify the dimensions
  TEST_ASSERT_EQUAL(rows, mat->rows);
  TEST_ASSERT_EQUAL(cols, mat->cols);

  // Verify the number of non-zero elements
  size_t expected_nnz = (size_t)(density * rows * cols);
  TEST_ASSERT_EQUAL(expected_nnz, mat->nnz);

  // Verify the values and positions
  for (size_t i = 0; i < mat->nnz; i++) {
    TEST_ASSERT_TRUE(mat->row_indices[i] < rows);
    TEST_ASSERT_TRUE(mat->col_indices[i] < cols);
    TEST_ASSERT_NOT_EQUAL(0.0, mat->values[i]);
  }

  // Verify the unique non-zero count matches expected density
  size_t unique_nnz = count_unique_non_zeros(mat);
  TEST_ASSERT_EQUAL(expected_nnz, unique_nnz);

  dms_destroy(mat);
}

void test_dms_rand_zero_density_returns_empty_matrix(void) {
  DoubleSparseMatrix *mat = dms_create_random(10, 12, 0.0);

  TEST_ASSERT_NOT_NULL(mat);
  TEST_ASSERT_EQUAL(10, mat->rows);
  TEST_ASSERT_EQUAL(12, mat->cols);
  TEST_ASSERT_EQUAL(0, mat->nnz);
  TEST_ASSERT_TRUE(mat->capacity >= 1);

  dms_destroy(mat);
}

void test_dms_convert_array(void) {
  double array[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  DoubleSparseMatrix *m = dms_create_from_2D_array(3, 3, array);
  TEST_ASSERT_EQUAL(3, m->rows);
  TEST_ASSERT_EQUAL(3, m->cols);
  TEST_ASSERT_EQUAL(9, m->nnz);
  TEST_ASSERT_EQUAL(10, m->capacity);
  dms_destroy(m);
}

void test_dms_get_row(void) {
  double array[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  DoubleSparseMatrix *m = dms_create_from_2D_array(3, 3, array);
  DoubleSparseMatrix *m2 = dms_get_row(m, 1);
  TEST_ASSERT_EQUAL(1, m2->rows);
  TEST_ASSERT_EQUAL(3, m2->cols);
  TEST_ASSERT_EQUAL(3, m2->nnz);
  TEST_ASSERT_EQUAL(4, m2->capacity);
  dms_destroy(m);
  dms_destroy(m2);
}

void test_dms_get_last_row(void) {
  double array[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  DoubleSparseMatrix *m = dms_create_from_2D_array(3, 3, array);
  DoubleSparseMatrix *m2 = dms_get_last_row(m);
  TEST_ASSERT_EQUAL(1, m2->rows);
  TEST_ASSERT_EQUAL(3, m2->cols);
  TEST_ASSERT_EQUAL(3, m2->nnz);
  TEST_ASSERT_EQUAL(4, m2->capacity);
  dms_destroy(m);
  dms_destroy(m2);
}

void test_dms_get_col(void) {
  double array[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  DoubleSparseMatrix *m = dms_create_from_2D_array(3, 3, array);
  DoubleSparseMatrix *m2 = dms_get_col(m, 1);
  TEST_ASSERT_EQUAL(3, m2->rows);
  TEST_ASSERT_EQUAL(1, m2->cols);
  TEST_ASSERT_EQUAL(3, m2->nnz);
  TEST_ASSERT_EQUAL(3, m2->capacity);
  dms_destroy(m);
  dms_destroy(m2);
}

void test_dms_get_last_col(void) {
  double array[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  DoubleSparseMatrix *m = dms_create_from_2D_array(3, 3, array);
  DoubleSparseMatrix *m2 = dms_get_last_col(m);
  TEST_ASSERT_EQUAL(3, m2->rows);
  TEST_ASSERT_EQUAL(1, m2->cols);
  TEST_ASSERT_EQUAL(3, m2->nnz);
  TEST_ASSERT_EQUAL(3, m2->capacity);
  dms_destroy(m);
  dms_destroy(m2);
}

// Test to check if transposition of a matrix is correct
void test_dms_transpose_basic(void) {
  size_t rows = 3, cols = 2;
  size_t nnz = 3;
  size_t row_indices[] = {0, 1, 2};
  size_t col_indices[] = {0, 1, 0};
  double values[] = {1.0, 2.0, 3.0};

  DoubleSparseMatrix *mat =
      dms_create_with_values(rows, cols, nnz, row_indices, col_indices, values);
  DoubleSparseMatrix *transposed = dms_transpose(mat);

  // Expected result matrix
  size_t expected_rows = 2, expected_cols = 3, expected_nnz = 3;
  size_t expected_row_indices[] = {0, 1, 0};
  size_t expected_col_indices[] = {0, 1, 2};
  double expected_values[] = {1.0, 2.0, 3.0};

  // Verify the dimensions
  TEST_ASSERT_EQUAL(expected_rows, transposed->rows);
  TEST_ASSERT_EQUAL(expected_cols, transposed->cols);

  // Verify the number of non-zero elements
  TEST_ASSERT_EQUAL(expected_nnz, transposed->nnz);

  // Verify the values and positions
  for (size_t i = 0; i < expected_nnz; i++) {
    TEST_ASSERT_EQUAL(expected_row_indices[i], transposed->row_indices[i]);
    TEST_ASSERT_EQUAL(expected_col_indices[i], transposed->col_indices[i]);
    TEST_ASSERT_EQUAL(expected_values[i], transposed->values[i]);
  }

  dms_destroy(mat);
  dms_destroy(transposed);
}

// Test to check if transposition of an empty matrix is handled correctly
// void test_dms_transpose_empty(void) {
//   size_t rows = 3, cols = 2;
//   size_t nnz = 0;

//   DoubleSparseMatrix *mat =
//       dms_create_with_values(rows, cols, nnz, NULL, NULL, NULL);
//   DoubleSparseMatrix *transposed = dms_transpose(mat);

//   // Verify the dimensions
//   TEST_ASSERT_EQUAL(cols, transposed->rows);
//   TEST_ASSERT_EQUAL(rows, transposed->cols);

//   // Verify the number of non-zero elements
//   TEST_ASSERT_EQUAL(nnz, transposed->nnz);

//   dms_destroy(mat);
//   dms_destroy(transposed);
// }

#if defined(TEST_DMS_HAS_CSPARSE)
void test_dms_to_cs(void) {
  double array[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  DoubleSparseMatrix *m = dms_create_from_2D_array(3, 3, array);
  cs *A = dms_to_cs(m);
  TEST_ASSERT_EQUAL(3, A->m);
  TEST_ASSERT_EQUAL(3, A->n);
  TEST_ASSERT_EQUAL(9, A->nzmax);
  dms_destroy(m);
  cs_spfree(A);
}

void test_cs_to_dms(void) {
  cs *A = cs_spalloc(3, 3, 9, 1, 1);
  A->p[0] = 0;
  A->p[1] = 3;
  A->p[2] = 6;
  A->p[3] = 9;
  A->i[0] = 0;
  A->i[1] = 1;
  A->i[2] = 2;
  A->i[3] = 0;
  A->i[4] = 1;
  A->i[5] = 2;
  A->i[6] = 0;
  A->i[7] = 1;
  A->i[8] = 2;
  A->x[0] = 1;
  A->x[1] = 2;
  A->x[2] = 3;
  A->x[3] = 4;
  A->x[4] = 5;
  A->x[5] = 6;
  A->x[6] = 7;
  A->x[7] = 8;
  A->x[8] = 9;
  DoubleSparseMatrix *m = cs_to_dms(A);
  TEST_ASSERT_EQUAL(3, m->rows);
  TEST_ASSERT_EQUAL(3, m->cols);
  TEST_ASSERT_EQUAL(9, m->nnz);
  TEST_ASSERT_EQUAL(109, m->capacity);
  dms_destroy(m);
  cs_spfree(A);
}
#else
void test_dms_to_cs(void) {
  TEST_IGNORE_MESSAGE("cs.h not available for editor-only parse context");
}

void test_cs_to_dms(void) {
  TEST_IGNORE_MESSAGE("cs.h not available for editor-only parse context");
}
#endif

// /******************************
//  ** Matrix operations:
//  *******************************/

// Test to check if multiplication of two matrices is correct
void test_dms_multiply_basic(void) {
  double array_A[2][3] = {{3.0, 2.0, 1.0}, {1.0, 0.0, 2.0}};
  DoubleSparseMatrix *m_A = dms_create_from_2D_array(2, 3, array_A);

  double array_B[3][2] = {{1.0, 2.0}, {0.0, 1.0}, {4.0, 0.0}};
  DoubleSparseMatrix *m_B = dms_create_from_2D_array(3, 2, array_B);

  DoubleSparseMatrix *result = dms_multiply(m_A, m_B);
  // Expected result matrix
  double expected_array[2][2] = {{7.0, 8.0}, {9.0, 2.0}};
  DoubleSparseMatrix *expected_result =
      dms_create_from_2D_array(2, 2, expected_array);

  // test if result is expected_result
  TEST_ASSERT_EQUAL(expected_result->rows, result->rows);
  TEST_ASSERT_EQUAL(expected_result->cols, result->cols);
  for (size_t i = 0; i < expected_result->rows; i++) {
    for (size_t j = 0; j < expected_result->cols; j++) {
      TEST_ASSERT_EQUAL(dms_get(expected_result, i, j), dms_get(result, i, j));
    }
  }
}

// Test to check multiplication with incompatible dimensions
void test_dms_multiply_incompatible_dimensions(void) {
  size_t rows1 = 2, cols1 = 3;
  size_t nnz1 = 3;
  size_t row_indices1[] = {0, 1, 1};
  size_t col_indices1[] = {0, 1, 2};
  double values1[] = {1.0, 2.0, 3.0};

  size_t rows2 = 2, cols2 = 2;
  size_t nnz2 = 2;
  size_t row_indices2[] = {0, 1};
  size_t col_indices2[] = {1, 0};
  double values2[] = {4.0, 5.0};

  DoubleSparseMatrix *mat1 = dms_create_with_values(
      rows1, cols1, nnz1, row_indices1, col_indices1, values1);
  DoubleSparseMatrix *mat2 = dms_create_with_values(
      rows2, cols2, nnz2, row_indices2, col_indices2, values2);

  DoubleSparseMatrix *result = dms_multiply(mat1, mat2);

  // Verify the result is NULL for incompatible dimensions
  TEST_ASSERT_NULL(result);

  dms_destroy(mat1);
  dms_destroy(mat2);
}

void test_dms_realloc(void) {
  DoubleSparseMatrix *m = dms_create(3, 3, 3);
  dms_realloc(m, 200);
  TEST_ASSERT_EQUAL(3, m->rows);
  TEST_ASSERT_EQUAL(3, m->cols);
  TEST_ASSERT_EQUAL(0, m->nnz);
  TEST_ASSERT_EQUAL(200, m->capacity);
  dms_destroy(m);
}

void test_dms_set(void) {
  DoubleSparseMatrix *m = dms_create(3, 3, 3);
  dms_set(m, 1, 1, 1);
  TEST_ASSERT_EQUAL(1, m->values[0]);
  dms_destroy(m);
}

void test_dms_get(void) {
  DoubleSparseMatrix *m = dms_create(3, 3, 3);
  dms_set(m, 1, 1, 1);
  double value = dms_get(m, 1, 1);
  TEST_ASSERT_EQUAL(1, value);
  dms_destroy(m);
}
