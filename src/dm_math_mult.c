/**
 * @file dm_math_mult.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 16-04-2023
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <assert.h>

#include "dm.h"
#include "dm_internals.h"
#include "dm_math.h"
#include "dm_math_blas.h"
#include "dm_vector.h"
#include <cblas.h>

/*******************************/
/*   Matrix Multiplication     */
/*******************************/

/**
 * @brief Matrix Multiplication of two matrices m1 x m2
 * @param m1
 * @param m2
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_multiply_by_matrix(const DoubleMatrix *mat1,
                                    const DoubleMatrix *mat2) {

  // check if matrices are empty
  if (mat1 == NULL || mat2 == NULL) {
    perror("Error: Matrices shouldn't be empty.");
    exit(EXIT_FAILURE);
  }
  // check if matrices have compatible dimensions
  if (mat1->cols != mat2->rows) {
    perror(
        "Error: number of columns of m1 has to be euqal to number of rows of "
        "m2!");
    exit(EXIT_FAILURE);
  }
  // check if matrices have the same format
  if (mat1->format != mat2->format) {
    perror("Error: Matrices have to be of the same format (DENSE, COO, CSR).");
    exit(EXIT_FAILURE);
  }

  switch (mat1->format) {
  case DENSE:
    return dm_blas_multiply_by_matrix(mat1, mat2);
    break;
  case COO:
    return dm_multiply_by_matrix_coo(mat1, mat2);
    break;
  case CSC:
    return NULL;
    break; // not implemented yet
  case VECTOR:
    return NULL;
    break; // not relevant
  }
}

// use cblas_dgemm to multiply two matrices
static DoubleMatrix *dm_blas_multiply_by_matrix(const DoubleMatrix *mat1,
                                                const DoubleMatrix *mat2) {

  DoubleMatrix *product = dm_create(mat1->rows, mat2->cols);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (blasint)mat1->rows,
              (blasint)mat2->cols, (blasint)mat1->cols, 1.0, mat1->values,
              (blasint)mat1->cols, mat2->values, (blasint)mat2->cols, 0.0,
              product->values, (blasint)product->cols);

  return product;
}

static DoubleMatrix *dm_multiply_by_matrix_coo(const DoubleMatrix *matrixA,
                                               const DoubleMatrix *matrixB) {

  DoubleMatrix *result = dm_create(matrixA->rows, matrixB->cols);

  size_t nnz_estimate =
      matrixA->nnz *
      matrixB->nnz; // Estimate for the non-zero elements in the result matrix

  if (nnz_estimate > result->capacity) {
    dm_realloc_coo(result, nnz_estimate);
  }
  result->nnz = 0;

  // Perform matrix multiplication
  for (size_t i = 0; i < matrixA->nnz; i++) {
    size_t rowA = matrixA->row_indices[i];
    size_t colA = matrixA->col_indices[i];
    double valueA = matrixA->values[i];

    for (size_t j = 0; j < matrixB->nnz; j++) {
      size_t rowB = matrixB->row_indices[j];
      size_t colB = matrixB->col_indices[j];
      double valueB = matrixB->values[j];

      if (colA == rowB) {
        // Perform the multiplication
        size_t rowResult = rowA;
        size_t colResult = colB;
        double valueResult = valueA * valueB;

        // Insert or accumulate the result in the output matrix
        accumulate_result(result, rowResult, colResult, valueResult);
      }
    }
  }

  return result;
}

static void accumulate_result(DoubleMatrix *result, size_t row, size_t col,
                              double value) {
  size_t position = binary_search_coo(result, row, col);

  if (position < result->nnz && result->row_indices[position] == row &&
      result->col_indices[position] == col) {
    // Element already exists at position (row, col), accumulate the value
    result->values[position] += value;
  } else {
    // Insert new element at the appropriate position
    insert_element(result, row, col, value, position);
  }
}

/*******************************/
/*      Naive Approach         */
/*******************************/

// static DoubleMatrix *dm_naive_multiply_by_matrix(const DoubleMatrix *mat1,
//                                                     const DoubleMatrix *mat2)
//                                                     {
//   DoubleMatrix *product = dm_create(mat1->rows, mat2->cols);

//   // Multiplying first and second matrices and storing it in product
//   for (size_t i = 0; i < mat1->rows; ++i) {
//     for (size_t j = 0; j < mat2->cols; ++j) {
//       for (size_t k = 0; k < mat1->cols; ++k) {
//         dm_set(product, i, j,
//                dm_get(product, i, j) + dm_get(mat1, i, k) * dm_get(mat2, k,
//                j));
//       }
//     }
//   }

//   return product;
// }

/*******************************/
/*       Matrix * scalar       */
/*******************************/

/**
 * @brief Multiply a matrix with a scalar
 *
 * @param mat
 * @param scalar
 * @return DoubleMatrix
 */
void dm_multiply_by_scalar(DoubleMatrix *mat, const double scalar) {
  switch (mat->format) {
  case DENSE:
    dm_multiply_by_scalar_dense(mat, scalar);
    break;
  case COO:
    dm_multiply_by_scalar_sparse(mat, scalar);
    break;
  case CSC:
    dm_multiply_by_scalar_sparse(mat, scalar);
    break;
  case VECTOR:
    dv_multiply_by_scalar(mat, scalar);
    break; // not relevant
  }
}

void dm_multiply_by_scalar_dense(DoubleMatrix *mat, const double scalar) {
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      mat->values[i * mat->cols + j] *= scalar;
    }
  }
}

static void dm_multiply_by_scalar_sparse(DoubleMatrix *mat,
                                         const double scalar) {
  for (size_t i = 0; i < mat->nnz; i++) {
    mat->values[i] *= scalar;
  }
}

/*******************************/
/*       Matrix * Vector       */
/*******************************/

/**
 * @brief Matrix-Vector Multiplication (n x m) x (n x 1)
 *
 * @param vec
 * @param mat
 * @return DoubleVector*
 */
DoubleVector *dm_multiply_by_vector(const DoubleMatrix *mat,
                                    const DoubleVector *vec) {
  if (vec->rows != mat->cols) {
    return NULL; // dimensions are incompatible, return NULL
  }

  DoubleVector *vec_result = dv_create(mat->rows);
  for (size_t i = 0; i < mat->rows; i++) {
    dv_set(vec_result, i, 0.0);
    for (size_t j = 0; j < vec->rows; j++) {
      dv_set(vec_result, i,
             dv_get(vec_result, i) + dm_get(mat, i, j) * dv_get(vec, j));
    }
  }
  return vec_result;
}
