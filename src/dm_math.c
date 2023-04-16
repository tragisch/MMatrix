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
#include "dm_matrix.h"

// #define NDEBUG
#define INITIAL_SIZE 4

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

/**
 * @brief check if two matrices are equal
 *
 * @param m1
 * @param m2
 * @return true
 * @return false
 */
bool dm_equal_matrix(DoubleMatrix *mat1, DoubleMatrix *mat2) {
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
DoubleMatrix *dm_multiply_with_matrix(DoubleMatrix *mat1, DoubleMatrix *mat2) {

  if (mat1 == NULL || mat2 == NULL) {
    perror("Error: Matrices shouldn't be empty.");
    return NULL;
  }

  if (mat1->cols != mat2->rows) {
    perror(
        "Error: number of columns of m1 has to be euqal to number fo rows of "
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
DoubleMatrix dm_multiply_by_scalar(const DoubleMatrix *mat,
                                   const double scalar) {
  DoubleMatrix *result = dm_create(mat->rows, mat->cols);
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(result, i, j, dm_get(mat, i, j) * scalar);
    }
  }
  return *result;
}

/**
 * @brief Vector Matrix
 *
 * @param vec
 * @param mat
 * @return DoubleVector*
 */
DoubleVector *dv_multiply_with_matrix(const DoubleVector *vec,
                                      const DoubleMatrix *mat) {
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
double dm_determinant(DoubleMatrix *mat) {
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
    det += pow(-1, i) * dm_get(mat, 0, i) * dm_determinant(sub_mat);
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
      dm_set(inverse, i, j, pow(-1, i + j) * dm_determinant(sub_mat));
      dm_destroy(sub_mat);
    }
  }
  dm_transpose(inverse);
  dm_multiply_by_scalar(inverse, 1 / det);
  return inverse;
}

/*******************************/
/*     Double Vector Math      */
/*******************************/

/**
 * @brief Multiply Vector v1 with Vectot v2  -- dot product!
 *
 * @param vec1
 * @param vec2
 * @return double
 */
double dv_dot_product(DoubleVector *vec1, DoubleVector *vec2) {
  if (vec1->rows != vec2->rows) {
    perror("vectors have not same length");
    return 0;
  }
  if (vec1->cols != vec2->cols) {
    perror("no dot product for: column-vectors * row_vector");
    return 0;
  }

  double sum = 0;
  for (size_t i = 0; i < vec1->rows; i++) {
    sum += dv_get(vec1, i) * dv_get(vec2, i);
  }

  return sum;
}

/**
 * @brief add Vector vec1 with Vector vec2
 *
 * @param vec1
 * @param vec2 (const)
 */
void dv_add_vector(DoubleVector *vec1, const DoubleVector *vec2) {
  if (vec1->rows != vec2->rows) {
    perror("vectors are not same length");
  }

  for (size_t i = 0; i < vec1->rows; i++) {
    dv_set(vec1, i, dv_get(vec1, i) + dv_get(vec2, i));
  }
}

/**
 * @brief sub Vector vec1 from Vector vec2 (vec1 - vec2)
 *
 * @param vec1
 * @param vec2 (const)
 */
void dv_sub_vector(DoubleVector *vec1, const DoubleVector *vec2) {
  if (vec1->rows != vec2->rows) {
    perror("vectors are not same length");
  }

  for (size_t i = 0; i < vec1->rows; i++) {
    dv_set(vec1, i, dv_get(vec1, i) - dv_get(vec2, i));
  }
}

/**
 * @brief multiply each element of Vector vec1 with a scalar
 *
 * @param vec
 * @param scalar
 */
void dv_multiply_by_scalar(DoubleVector *vec, const double scalar) {
  for (size_t i = 0; i < vec->rows; i++) {
    dv_set(vec, i, dv_get(vec, i) * scalar);
  }
}

/**
 * @brief divied each element of Vector vec1 with a scalar
 *
 * @param vec
 * @param scalar
 */
void dv_divide_by_scalar(DoubleVector *vec, const double scalar) {
  for (size_t i = 0; i < vec->rows; i++) {
    dv_set(vec, i, dv_get(vec, i) / scalar);
  }
}

/**
 * @brief add constant to vector
 *
 * @param vec
 * @param scalar
 */
void dv_add_constant(DoubleVector *vec, const double constant) {
  for (size_t i = 0; i < vec->rows; i++) {
    dv_set(vec, i, dv_get(vec, i) + constant);
  }
}

/**
 * @brief test if two DoubleVectors are equal
 *
 * @param vec1
 * @param vec2
 * @return bool
 */
bool dv_equal(DoubleVector *vec1, DoubleVector *vec2) {
  if (vec1->rows != vec2->rows) {
    return false;
  }
  for (size_t i = 0; i < vec1->rows; i++) {
    if (dv_get(vec1, i) != dv_get(vec2, i)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief return mean of Vector vec
 *
 * @param vec
 * @return double
 */
double dv_mean(DoubleVector *vec) {
  double mean = 0.0;
  for (size_t i = 0; i < vec->rows; i++) {
    mean += dv_get(vec, i);
  }
  return (mean / (double)vec->rows);
}

/**
 * @brief return min of Vector vec
 *
 * @param vec
 * @return double
 */
double dv_min(DoubleVector *vec) {
  double min = vec->values[0];
  for (size_t i = 1; i < vec->rows; i++) {
    if (min > vec->values[i]) {
      min = vec->values[i];
    }
  }
  return min;
}

/**
 * @brief return max of Vector vec
 *
 * @param vec
 * @return double
 */
double dv_max(DoubleVector *vec) {
  double max = vec->values[0];
  for (size_t i = 1; i < vec->rows; i++) {
    if (max < vec->values[i]) {
      max = vec->values[i];
    }
  }
  return max;
}

/**
 * @brief transpose a vector
 *
 * @param vec*
 */
void dv_transpose(DoubleVector *vec) {
  if (vec->rows == 1) {
    vec->rows = vec->cols;
    vec->cols = 1;
  } else {
    vec->cols = vec->rows;
    vec->rows = 1;
  }
}

/**
 * @brief swap two elements of an vector
 *
 * @param vec*
 * @param i
 * @param j
 */
void dv_swap_elements(DoubleVector *vec, size_t idx_i, size_t idx_j) {
  double tmp = dv_get(vec, idx_i);
  dv_set(vec, idx_i, dv_get(vec, idx_j));
  dv_set(vec, idx_j, tmp);
}

/**
 * @brief reverse the order of elements of vec
 *
 * @param vec*
 */
void dv_reverse(DoubleVector *vec) {
  // reverse the order of elements of vec
  for (size_t i = 0; i < vec->rows / 2; i++) {
    dv_swap_elements(vec, i, vec->rows - i - 1);
  }
}

/**
 * @brief return the magnitude of a vector
 *
 * @param vec
 * @return double
 */
double dv_magnitude(DoubleVector *vec) {
  double sum_of_squares = 0.0;
  for (size_t i = 0; i < vec->rows; i++) {
    double component = dv_get(vec, i);
    sum_of_squares += component * component;
  }
  double magnitude = sqrt(sum_of_squares);
  return magnitude;
}

/**
 * @brief normalize a vector
 *
 * @param vec
 */
void dv_normalize(DoubleVector *vec) {
  double magnitude = dv_magnitude(vec);
  dv_divide_by_scalar(vec, magnitude);
}
