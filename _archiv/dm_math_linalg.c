/**
 * @file dm_math_linalg.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 16-04-2023
 *
 * @copyright Copyright (c) 2021
 *
 */

// #include <assert.h>

#include "dm.h"
#include "dm_internals.h"
#include "dm_math.h"

/*******************************/
/*  Using Apples Accelarator   */
/*******************************/

#ifdef APPLE_BLAS
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/*******************************/
/*         Determinant         */
/*******************************/

/**
 * @brief Matrix Vector
 *
 * @param mat
 * @return value
 */
double dm_determinant(const DoubleMatrix *mat) {
  if (mat->cols != mat->rows) {
    perror("the Matrix has to be square!");
  }
  if (mat->cols == 1) {
    return dm_get(mat, 0, 0);
  }
  if (mat->cols == 2) {
    return dm_get(mat, 0, 0) * dm_get(mat, 1, 1) -
           dm_get(mat, 0, 1) * dm_get(mat, 1, 0);
  }

  double det = 0;
  for (size_t i = 0; i < mat->cols; i++) {
    DoubleMatrix *sub_mat = dm_create(mat->cols - 1, mat->cols - 1);
    for (size_t j = 1; j < mat->cols; j++) {
      for (size_t k = 0; k < mat->cols; k++) {
        if (k < i) {
          dm_set(sub_mat, j - 1, k, dm_get(mat, j, k));
        } else if (k > i) {
          dm_set(sub_mat, j - 1, k - 1, dm_get(mat, j, k));
        }
      }
    }
    det += pow(-1, (double)i) * dm_get(mat, 0, i) * dm_determinant(sub_mat);
    dm_destroy(sub_mat);
  }
  return det;
}

/*******************************/
/*       Inverse Matrix        */
/*******************************/

/**
 * @brief dm_inverse
 *
 * @param mat
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_inverse(DoubleMatrix *mat) {
  if (mat->cols != mat->rows) {
    perror("the Matrix has to be square!");
  }
  double det = dm_determinant(mat);
  if (det == 0) {
    perror("the Matrix has no inverse!");
    return NULL;
  }
  DoubleMatrix *inverse = dm_create(mat->cols, mat->cols);
  for (size_t i = 0; i < mat->cols; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      DoubleMatrix *sub_mat = dm_create(mat->cols - 1, mat->cols - 1);
      for (size_t k = 0; k < mat->cols; k++) {
        for (size_t l = 0; l < mat->cols; l++) {
          if (k < i && l < j) {
            dm_set(sub_mat, k, l, dm_get(mat, k, l));
          } else if (k < i && l > j) {
            dm_set(sub_mat, k, l - 1, dm_get(mat, k, l));
          } else if (k > i && l < j) {
            dm_set(sub_mat, k - 1, l, dm_get(mat, k, l));
          } else if (k > i && l > j) {
            dm_set(sub_mat, k - 1, l - 1, dm_get(mat, k, l));
          }
        }
      }
      dm_set(inverse, i, j, pow(-1, (double)(i + j)) * dm_determinant(sub_mat));
      dm_destroy(sub_mat);
    }
  }
  dm_transpose(inverse);
  dm_multiply_by_scalar(inverse, 1 / det);
  return inverse;
}

/*******************************/
/*       Rank of Matrix        */
/*******************************/

/**
 * @brief returns the rank of a matrix
 *
 * @param mat
 * @return size_t
 */
size_t dm_rank(const DoubleMatrix *mat) {
  // Make a copy of the matrix to preserve the original data
  DoubleMatrix *copy = dm_clone(mat);

  // Apply Gaussian Elimination on the copy
  dm_gauss_elimination(copy);

  // Count the number of non-zero rows in the row-echelon form
  size_t rank = 0;
  for (size_t i = 0; i < copy->rows; i++) {
    int non_zero = 0;
    for (size_t j = 0; j < copy->cols; j++) {
      if (dm_get(copy, i, j) != 0.0) {
        non_zero = 1;
        break;
      }
    }
    if (non_zero) {
      rank++;
    }
  }

  // Free the memory of the copy
  dm_destroy(copy);

  return rank;
}

// Gaussian Elimination
static void dm_gauss_elimination(DoubleMatrix *mat) {
  size_t rows = mat->rows;
  size_t cols = mat->cols;

  // Apply Gaussian Elimination
  for (size_t pivot = 0; pivot < rows; pivot++) {
    // Find the maximum value in the column below the pivot
    double max_val = fabs(dm_get(mat, pivot, pivot));
    size_t max_row = pivot;
    for (size_t row = pivot + 1; row < rows; row++) {
      double val = fabs(dm_get(mat, row, pivot));
      if (val > max_val) {
        max_val = val;
        max_row = row;
      }
    }

    // Swap rows if necessary
    if (max_row != pivot) {
      for (size_t col = 0; col < cols; col++) {
        double temp = dm_get(mat, pivot, col);
        dm_set(mat, pivot, col, dm_get(mat, max_row, col));
        dm_set(mat, max_row, col, temp);
      }
    }

    // Perform row operations to eliminate values below the pivot
    for (size_t row = pivot + 1; row < rows; row++) {
      double factor = dm_get(mat, row, pivot) / dm_get(mat, pivot, pivot);
      dm_set(mat, row, pivot, 0.0); // Eliminate the value below the pivot

      for (size_t col = pivot + 1; col < cols; col++) {
        double val = dm_get(mat, row, col);
        double pivot_val = dm_get(mat, pivot, col);
        dm_set(mat, row, col, val - factor * pivot_val);
      }
    }
  }
}
