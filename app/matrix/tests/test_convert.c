/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "m_convert.h"

/******************************
 ** Test preconditions:
 *******************************/
#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10

/* Support for Meta Test Rig */
// #define TEST_CASE(...)

#include "unity.h"
#include "unity_internals.h"

#ifndef INIT_CAPACITY
#define INIT_CAPACITY 100
#endif
#ifndef EPSILON
#define EPSILON 1e-9
#endif

void test_convert_dense_matrix(void) {
  // Erstelle eine 3x3 Dense-Matrix
  double values[3][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  DoubleMatrix *dm = dm_from_array_static(3, 3, values);

  // Konvertiere zu Sparse-Matrix
  DoubleSparseMatrix *dms = dm_to_dms(dm);
  TEST_ASSERT_NOT_NULL(dms);
  TEST_ASSERT_EQUAL(3, dms->rows);
  TEST_ASSERT_EQUAL(3, dms->cols);
  TEST_ASSERT_EQUAL(9, dms->nnz); // Alle Werte sind ungleich 0

  // Konvertiere zurück zu Dense-Matrix
  DoubleMatrix *dm_converted = dms_to_dm(dms);
  TEST_ASSERT_NOT_NULL(dm_converted);
  TEST_ASSERT_EQUAL(3, dm_converted->rows);
  TEST_ASSERT_EQUAL(3, dm_converted->cols);

  // Überprüfe, ob die Werte übereinstimmen
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      TEST_ASSERT_DOUBLE_WITHIN(EPSILON, values[i][j],
                                dm_get(dm_converted, i, j));
    }
  }

  // Speicher freigeben
  dm_destroy(dm);
  dms_destroy(dms);
  dm_destroy(dm_converted);
}

void test_convert_matrix_with_zeros(void) {
  // Erstelle eine 3x3 Dense-Matrix mit Nullwerten
  double values[3][3] = {{1.0, 0.0, 3.0}, {0.0, 5.0, 0.0}, {7.0, 0.0, 9.0}};
  DoubleMatrix *dm = dm_from_array_static(3, 3, values);

  // Konvertiere zu Sparse-Matrix
  DoubleSparseMatrix *dms = dm_to_dms(dm);
  TEST_ASSERT_NOT_NULL(dms);
  TEST_ASSERT_EQUAL(3, dms->rows);
  TEST_ASSERT_EQUAL(3, dms->cols);
  TEST_ASSERT_EQUAL(5, dms->nnz); // Nur 5 Werte sind ungleich 0

  // Konvertiere zurück zu Dense-Matrix
  DoubleMatrix *dm_converted = dms_to_dm(dms);
  TEST_ASSERT_NOT_NULL(dm_converted);
  TEST_ASSERT_EQUAL(3, dm_converted->rows);
  TEST_ASSERT_EQUAL(3, dm_converted->cols);

  // Überprüfe, ob die Werte übereinstimmen
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      TEST_ASSERT_DOUBLE_WITHIN(EPSILON, values[i][j],
                                dm_get(dm_converted, i, j));
    }
  }

  // Speicher freigeben
  dm_destroy(dm);
  dms_destroy(dms);
  dm_destroy(dm_converted);
}

void test_convert_large_matrix(void) {
  // Erstelle eine große Sparse-Matrix (1000x1000) mit wenigen nicht-null Werten
  size_t rows = 1000, cols = 1000, nnz = 10;
  size_t row_indices[10] = {0, 100, 200, 300, 400, 500, 600, 700, 800, 900};
  size_t col_indices[10] = {0, 100, 200, 300, 400, 500, 600, 700, 800, 900};
  double values[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  DoubleSparseMatrix *dms =
      dms_create_with_values(rows, cols, nnz, row_indices, col_indices, values);

  // Konvertiere zu Dense-Matrix
  DoubleMatrix *dm = dms_to_dm(dms);
  TEST_ASSERT_NOT_NULL(dm);
  TEST_ASSERT_EQUAL(rows, dm->rows);
  TEST_ASSERT_EQUAL(cols, dm->cols);

  // Überprüfe einige Werte
  TEST_ASSERT_DOUBLE_WITHIN(EPSILON, 1.0, dm_get(dm, 0, 0));
  TEST_ASSERT_DOUBLE_WITHIN(EPSILON, 10.0, dm_get(dm, 900, 900));
  TEST_ASSERT_DOUBLE_WITHIN(EPSILON, 0.0, dm_get(dm, 500, 501)); // Nullwert

  // Speicher freigeben
  dms_destroy(dms);
  dm_destroy(dm);
}

void test_convert_sparse_matrix(void) {
  // Erstelle eine Sparse-Matrix mit 3x3 und 3 nicht-null Werten
  size_t row_indices[3] = {0, 1, 2};
  size_t col_indices[3] = {0, 1, 2};
  double values[3] = {1.0, 5.0, 9.0};
  DoubleSparseMatrix *dms =
      dms_create_with_values(3, 3, 3, row_indices, col_indices, values);

  // Konvertiere zu Dense-Matrix
  DoubleMatrix *dm = dms_to_dm(dms);
  TEST_ASSERT_NOT_NULL(dm);
  TEST_ASSERT_EQUAL(3, dm->rows);
  TEST_ASSERT_EQUAL(3, dm->cols);

  // Überprüfe, ob die Werte korrekt sind
  TEST_ASSERT_DOUBLE_WITHIN(EPSILON, 1.0, dm_get(dm, 0, 0));
  TEST_ASSERT_DOUBLE_WITHIN(EPSILON, 5.0, dm_get(dm, 1, 1));
  TEST_ASSERT_DOUBLE_WITHIN(EPSILON, 9.0, dm_get(dm, 2, 2));
  TEST_ASSERT_DOUBLE_WITHIN(EPSILON, 0.0, dm_get(dm, 0, 1)); // Nullwerte

  // Konvertiere zurück zu Sparse-Matrix
  DoubleSparseMatrix *dms_converted = dm_to_dms(dm);
  TEST_ASSERT_NOT_NULL(dms_converted);
  TEST_ASSERT_EQUAL(3, dms_converted->rows);
  TEST_ASSERT_EQUAL(3, dms_converted->cols);
  TEST_ASSERT_EQUAL(3, dms_converted->nnz);

  // Speicher freigeben
  dms_destroy(dms);
  dm_destroy(dm);
  dms_destroy(dms_converted);
}
