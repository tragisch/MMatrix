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

void test_convert_st_from_sm_should_copy_shape_and_values(void) {
  float values[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  FloatMatrix *sm = sm_from_array_static(2, 3, values);
  TEST_ASSERT_NOT_NULL(sm);

  FloatTensor *st = st_from_sm(sm);
  TEST_ASSERT_NOT_NULL(st);
  TEST_ASSERT_EQUAL(2, st->ndim);
  TEST_ASSERT_EQUAL(2, st->shape[0]);
  TEST_ASSERT_EQUAL(3, st->shape[1]);
  TEST_ASSERT_EQUAL(6, st->numel);

  float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  for (size_t i = 0; i < 6; ++i) {
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, expected[i], st->values[i]);
  }

  // Ensure conversion is a deep copy
  sm->values[0] = 99.0f;
  TEST_ASSERT_DOUBLE_WITHIN(1e-6, 1.0f, st->values[0]);

  sm_destroy(sm);
  st_destroy(st);
}

void test_convert_sm_from_st_should_copy_shape_and_values(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *st = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(st);

  float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  for (size_t i = 0; i < 6; ++i) {
    st->values[i] = values[i];
  }

  FloatMatrix *sm = sm_from_st(st);
  TEST_ASSERT_NOT_NULL(sm);
  TEST_ASSERT_EQUAL(2, sm->rows);
  TEST_ASSERT_EQUAL(3, sm->cols);

  for (size_t i = 0; i < 6; ++i) {
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, values[i], sm->values[i]);
  }

  // Ensure conversion is a deep copy
  st->values[0] = 99.0f;
  TEST_ASSERT_DOUBLE_WITHIN(1e-6, 1.0f, sm->values[0]);

  sm_destroy(sm);
  st_destroy(st);
}

void test_convert_sm_from_st_should_materialize_non_contiguous_2d_view(void) {
  size_t shape[2] = {2, 3};
  FloatTensor *base = st_create(2, shape);
  TEST_ASSERT_NOT_NULL(base);

  float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  for (size_t i = 0; i < 6; ++i) {
    base->values[i] = values[i];
  }

  size_t perm[2] = {1, 0};
  FloatTensor *view = st_permute_view(base, perm);
  TEST_ASSERT_NOT_NULL(view);

  FloatMatrix *sm = sm_from_st(view);
  TEST_ASSERT_NOT_NULL(sm);
  TEST_ASSERT_EQUAL(3, sm->rows);
  TEST_ASSERT_EQUAL(2, sm->cols);

  float expected[3][2] = {
      {1.0f, 4.0f},
      {2.0f, 5.0f},
      {3.0f, 6.0f},
  };

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      TEST_ASSERT_DOUBLE_WITHIN(1e-6, expected[i][j], sm_get(sm, i, j));
    }
  }

  sm_destroy(sm);
  st_destroy(view);
  st_destroy(base);
}
