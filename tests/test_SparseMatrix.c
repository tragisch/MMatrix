#include "dm_math.h"
#include "dm_matrix.h"
#include "sp_math.h"
#include "sp_matrix.h"
#include <stddef.h>
#include <stdlib.h>

/******************************
 ** Test preconditions:
 *******************************/
#define UNITY_INCLUDE_DOUBLE
#define UNITY_DOUBLE_PRECISION 10

#include "unity.h"
#include "unity_internals.h"

/******************************
 ** Tests
 *******************************/

void test_sp_convert_coo_to_csr() {
  // Set up a COO-format sparse matrix
  SparseMatrix *coo = (SparseMatrix *)malloc(sizeof(SparseMatrix));
  coo->rows = 3;
  coo->cols = 3;
  coo->nnz = 4;
  coo->format = COO;
  coo->row_indices = (size_t *)malloc(coo->nnz * sizeof(size_t));
  coo->col_indices = (size_t *)malloc(coo->nnz * sizeof(size_t));
  coo->values = (double *)malloc(coo->nnz * sizeof(double));
  coo->row_indices[0] = 0;
  coo->col_indices[0] = 0;
  coo->values[0] = 1.0;
  coo->row_indices[1] = 0;
  coo->col_indices[1] = 2;
  coo->values[1] = 2.0;
  coo->row_indices[2] = 1;
  coo->col_indices[2] = 1;
  coo->values[2] = 3.0;
  coo->row_indices[3] = 2;
  coo->col_indices[3] = 2;
  coo->values[3] = 4.0;

  // Convert COO to CSR
  sp_convert_to_csr(coo);

  // Check the CSR-format sparse matrix
  TEST_ASSERT_EQUAL(CSR, coo->format);
  TEST_ASSERT_EQUAL_UINT(3, coo->rows);
  TEST_ASSERT_EQUAL_UINT(3, coo->cols);
  TEST_ASSERT_EQUAL_UINT(4, coo->nnz);
  TEST_ASSERT_EQUAL_FLOAT(1.0, coo->values[0]);
  TEST_ASSERT_EQUAL_FLOAT(2.0, coo->values[1]);
  TEST_ASSERT_EQUAL_FLOAT(3.0, coo->values[2]);
  TEST_ASSERT_EQUAL_FLOAT(4.0, coo->values[3]);
  TEST_ASSERT_EQUAL_UINT(0, coo->row_indices[0]);
  TEST_ASSERT_EQUAL_UINT(2, coo->row_indices[1]);
  TEST_ASSERT_EQUAL_UINT(3, coo->row_indices[2]);
  TEST_ASSERT_EQUAL_UINT(0, coo->col_indices[0]);
  TEST_ASSERT_EQUAL_UINT(2, coo->col_indices[1]);
  TEST_ASSERT_EQUAL_UINT(1, coo->col_indices[2]);
  TEST_ASSERT_EQUAL_UINT(2, coo->col_indices[3]);

  // Free memory
  sp_destroy(coo);
}
