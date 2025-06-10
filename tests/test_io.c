/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "dm.h"
#include "dms.h"
#include "m_io.h"
#include "sm.h"

/******************************
 ** Test preconditions:
 *******************************/
#define UNITY_INCLUDE_FLOAT
#define UNITY_FLOAT_PRECISION 5

/* Support for Meta Test Rig */
#define TEST_CASE(...)

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Creation of matrices:
 *******************************/

#define EPSILON 1e-5f

void setUp(void) {
  // Remove the test file if it exists
  // remove("test_matrix.mat");
}

// Teardown function called after each test
void tearDown(void) {
  // Remove the test file if it exists
  // remove("test_matrix.mat");
}

DoubleMatrix *dm_create_sample_matrix(size_t rows, size_t cols) {
  DoubleMatrix *matrix = dm_create(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      dm_set(matrix, i, j, (double)(i * cols + j));
    }
  }
  return matrix;
}

FloatMatrix *sm_create_sample_matrix(size_t rows, size_t cols) {
  FloatMatrix *matrix = sm_create(rows, cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      sm_set(matrix, i, j, (double)(i * cols + j));
    }
  }
  return matrix;
}

// Test function for sm_write_to_file
void test_sm_write_to_file(void) {
  FloatMatrix *matrix = sm_create_sample_matrix(10, 10);
  const char *filename = "test_matrix.mat";

  // Write the matrix to a file
  int result = sm_write_MAT_file(matrix, filename);
  TEST_ASSERT_EQUAL(0, result);

  // Clean up
  sm_destroy(matrix);

  // Check if the file was created
  FILE *file = fopen(filename, "r");
  TEST_ASSERT_NOT_NULL(file);
  if (file) {
    fclose(file);
  }
}

// // Test function for sm_read_from_file
void test_sm_read_from_file(void) {
  const char *filename = "test_matrix.mat";

  // Read the matrix from the file
  FloatMatrix *matrix = sm_read_MAT_file(filename);
  TEST_ASSERT_NOT_NULL(matrix);

  if (matrix) {
    // Check the matrix dimensions
    TEST_ASSERT_EQUAL(10, matrix->rows);
    TEST_ASSERT_EQUAL(10, matrix->cols);

    // Verify the matrix values
    for (size_t i = 0; i < 10; ++i) {
      for (size_t j = 0; j < 10; ++j) {
        float expected_value = (float)(i * 10 + j);
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected_value, sm_get(matrix, i, j));
      }
    }

    // Clean up
    sm_destroy(matrix);
  }
}

// Test function for dm_write_to_file
void test_dm_write_to_file(void) {
  DoubleMatrix *matrix = dm_create_sample_matrix(10, 10);
  const char *filename = "test_matrix.mat";

  // Write the matrix to a file
  int result = dm_write_MAT_file(matrix, filename);
  TEST_ASSERT_EQUAL(0, result);

  // Clean up
  dm_destroy(matrix);

  // Check if the file was created
  FILE *file = fopen(filename, "r");
  TEST_ASSERT_NOT_NULL(file);
  if (file) {
    fclose(file);
  }
}

// Test function for dm_read_from_file
void test_dm_read_from_file(void) {
  const char *filename = "test_matrix.mat";

  // Read the matrix from the file
  DoubleMatrix *matrix = dm_read_MAT_file(filename);
  TEST_ASSERT_NOT_NULL(matrix);

  if (matrix) {
    // Check the matrix dimensions
    TEST_ASSERT_EQUAL(10, matrix->rows);
    TEST_ASSERT_EQUAL(10, matrix->cols);

    // Verify the matrix values
    for (size_t i = 0; i < 10; ++i) {
      for (size_t j = 0; j < 10; ++j) {
        double expected_value = (double)(i * 10 + j);
        TEST_ASSERT_EQUAL_FLOAT(expected_value, dm_get(matrix, i, j));
      }
    }

    // Clean up
    dm_destroy(matrix);
  }
}

// import Matlab MAT file and plot the matrix
void test_import_and_plot_matrix(void) {
  const char *filename = "tests/test_data/rand10.mat";

  // Read the matrix from the file
  DoubleMatrix *matrix = dm_read_MAT_file(filename);
  TEST_ASSERT_NOT_NULL(matrix);

  if (matrix) {
    // Check the matrix dimensions
    TEST_ASSERT_EQUAL(10, matrix->rows);
    TEST_ASSERT_EQUAL(10, matrix->cols);

    // Plot the matrix
    dm_print(matrix);

    // Clean up
    dm_destroy(matrix);
  }
}

extern char grid[HEIGHT][WIDTH];

void test_dm_cplot_grid_output(void) {
  DoubleMatrix *matrix = dm_create(5, 5);
  dm_set(matrix, 1, 2, 1.0);  // einzelner Punkt
  dm_cplot(matrix);

  // Prüfe ob das erwartete Zeichen im Grid gesetzt wurde
  // int x = get_x_coord(1, 5);
  // int y = get_y_coord(2, 5);
  // TEST_ASSERT_EQUAL_CHAR('*', grid[y][x]);

  dm_destroy(matrix);
}