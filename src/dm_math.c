/**
 * @file dm_math.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 16-04-2023
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <assert.h>

#include "dbg.h"
#include "dm.h"
#include "dm_internals.h"
#include "dm_math.h"
#include "dv_vector.h"
#include <float.h>

/*******************************/
/*   Random functions          */
/*******************************/

int max_int(int a, int b) { return a > b ? a : b; }
double max_double(double a, double b) { return a > b ? a : b; }

double randomDouble() {
  uint32_t random_uint32 = randomInt();
  double random_double = (double)random_uint32 / (double)UINT32_MAX;
  return random_double;
}
double randomDouble_betweenBounds(uint32_t min, uint32_t max) {
  return (randomInt_betweenBounds(min, max - 1) + randomDouble());
}

uint32_t randomInt() { return random_number_generator(); }

#ifdef __APPLE__

uint32_t randomInt_upperBound(uint32_t limit) {
  return arc4random_uniform(limit);
}

#else

uint32_t randomInt_upperBound(uint32_t limit) {
  static int initialized = 0;
  if (!initialized) {
    srand(time(NULL));
    initialized = 1;
  }
  int r;
  do {
    r = rand();
  } while (r >= RAND_MAX - RAND_MAX % limit);
  return r % limit;
}

#endif

uint32_t randomInt_betweenBounds(uint32_t min, uint32_t max) {
  if (max < min) {
    return min;
  }
  return randomInt_upperBound((max - min) + 1) + min;
}

bool is_zero(double value) { return fabs(value) < DBL_EPSILON; }

/*******************************/
/*     Double Matrix Math      */
/*******************************/

/**
 * @brief dm_transpose matrix mat
 *
 * @param mat
 * @return mat
 */
void dm_transpose(DoubleMatrix *mat) {
  if (mat->cols != mat->rows) {
    perror("the Matrix has to be square!");
  }
  double temp = 0.0;
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = i + 1; j < mat->cols; j++) {
      temp = dm_get(mat, i, j);
      dm_set(mat, i, j, dm_get(mat, j, i));
      dm_set(mat, j, i, temp);
    }
  }
}

static double sp_density(const DoubleMatrix *mat) {
  return ((double)mat->nnz / (double)(mat->rows * mat->cols));
}

/**
 * @brief check if two matrices are equal
 *
 * @param m1
 * @param m2
 * @return true
 * @return false
 */
bool dm_equal_matrix(const DoubleMatrix *mat1, const DoubleMatrix *mat2) {
  if (mat1 == NULL || mat2 == NULL) {
    return false;
  }
  if (mat1->cols != mat2->cols || mat1->rows != mat2->rows) {
    return false;
  }
  for (size_t i = 0; i < mat1->cols * mat1->rows; i++) {
    if (mat1->values[i] != mat2->values[i]) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Matrix Multiplication of two matrices m1 x m2
 *
 * @param m1
 * @param m2
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_multiply_by_matrix(const DoubleMatrix *mat1,
                                    const DoubleMatrix *mat2) {

  if (mat1 == NULL || mat2 == NULL) {
    perror("Error: Matrices shouldn't be empty.");
    return NULL;
  }

  if (mat1->cols != mat2->rows) {
    perror(
        "Error: number of columns of m1 has to be euqal to number of rows of "
        "m2!");
    return NULL;
  }

  DoubleMatrix *product = dm_create(mat1->rows, mat2->cols);

  // Multiplying first and second matrices and storing it in product
  for (size_t i = 0; i < mat1->rows; ++i) {
    for (size_t j = 0; j < mat2->cols; ++j) {
      for (size_t k = 0; k < mat1->cols; ++k) {
        dm_set(product, i, j,
               dm_get(product, i, j) + dm_get(mat1, i, k) * dm_get(mat2, k, j));
      }
    }
  }

  return product;
}

/**
 * @brief Multiply a matrix with a scalar
 *
 * @param mat
 * @param scalar
 * @return DoubleMatrix
 */
void dm_multiply_by_scalar(DoubleMatrix *mat, const double scalar) {
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, dm_get(mat, i, j) * scalar);
    }
  }
}

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

/**
 * @brief Matrix Vector
 *
 * @param mat
 * @param vec
 * @return DoubleVector*
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

/**
 * @brief returns the trace of a matrix
 *
 * @param mat
 * @return double
 */
double dm_trace(const DoubleMatrix *mat) {
  double trace = 0;
  for (size_t i = 0; i < mat->rows; i++) {
    trace += dm_get(mat, i, i);
  }
  return trace;
}

/**
 * @brief returns the density of a matrix
 *
 * @param mat
 * @return double
 */
double dm_density(const DoubleMatrix *mat) {
  if (mat->format != DENSE) {
    return sp_density(mat);
  }

  double density = 0.0;
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      if (dm_get(mat, i, j) != 0.0) {
        density += 1.0;
      }
    }
  }
  return (density / (double)(mat->rows * mat->cols));
}

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
