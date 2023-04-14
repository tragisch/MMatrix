/**
 * @file dm_math.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
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
 * @brief multiply scalar with each cell of matrix
 *
 * @param mat
 * @param scalar
 */
void dm_multiply_by_scalar(DoubleMatrix *mat, const double scalar) {
  for (size_t i = 0; i < mat->columns; i++) {
    for (size_t j = 0; j < mat->rows; j++) {
      (mat->values[i])[j] *= scalar;
    }
  }
}

/**
 * @brief dm_transpose matrix mat
 *
 * @param mat
 * @return mat
 */
void dm_transpose(DoubleMatrix *mat) {
  if (mat->columns != mat->rows) {
    perror("the Matrix has to be square!");
  }
  double temp = 0.0;
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = i + 1; j < mat->columns; j++) {
      temp = mat->values[i][j];
      mat->values[i][j] = mat->values[j][i];
      mat->values[j][i] = temp;
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
  if (mat1->columns != mat2->columns || mat1->rows != mat2->rows) {
    return false;
  }
  for (size_t i = 0; i < mat1->columns; i++) {
    for (size_t j = 0; j < mat1->rows; j++) {
      if (mat1->values[i][j] != mat2->values[i][j]) {
        return false;
      }
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

  if (mat1->columns != mat2->rows) {
    perror(
        "Error: number of columns of m1 has to be euqal to number fo rows of "
        "m2!");
    return NULL;
  }

  DoubleMatrix *product = dm_create(mat1->rows, mat2->columns);

  // Multiplying first and second matrices and storing it in product
  for (size_t i = 0; i < mat1->rows; ++i) {
    for (size_t j = 0; j < mat2->columns; ++j) {
      for (size_t k = 0; k < mat1->columns; ++k) {
        product->values[i][j] += mat1->values[i][k] * mat2->values[k][j];
      }
    }
  }

  return product;
}

// /* v1 x v2  -- simply a helper function -- computes dot product between two
//  * vectors*/
// static double vector_multiply(const double *col, const double *row,
//                               size_t length) {
//   double sum = 0.;
//   for (size_t i = 0; i < length; i++) {
//     sum += col[i] * row[i];
//   }
//   return sum;
// }

/**
 * @brief Vector Matrix
 *
 * @param vec
 * @param mat
 * @return DoubleVector*
 */
// DoubleVector *dv_multiply_with_matrix(const DoubleVector *vec,
//                                       const DoubleMatrix *mat) {
//   DoubleVector *vec_result = dv_create(vec->length);
//   for (size_t i = 0; i < vec->length; i++) {
//     vec_result->mat1D->values[i][0] = vector_multiply(
//         mat->values[i], (double *)vec->mat1D->values, vec->length);
//   }

//   return vec_result;
// }

DoubleVector *dv_multiply_with_matrix(const DoubleVector *vec,
                                      const DoubleMatrix *mat) {
  if (vec->length != mat->columns) {
    return NULL; // dimensions are incompatible, return NULL
  }

  DoubleVector *vec_result = dv_create(mat->rows);
  for (size_t i = 0; i < mat->rows; i++) {
    vec_result->mat1D->values[i][0] = 0.0;
    for (size_t j = 0; j < vec->length; j++) {
      vec_result->mat1D->values[i][0] +=
          mat->values[i][j] * vec->mat1D->values[j][0];
    }
  }
  return vec_result;
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
  if (vec1->length != vec2->length) {
    perror("vectors have not same length");
    return 0;
  }
  if (vec1->isColumnVector != vec2->isColumnVector) {
    perror("no dot product for: column-vectors * row_vector");
    return 0;
  }

  double sum = 0;
  for (size_t i = 0; i < vec1->length; i++) {
    sum += vec1->mat1D->values[i][0] * vec2->mat1D->values[i][0];
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
  if (vec1->length != vec2->length) {
    perror("vectors are not same length");
  }

  for (size_t i = 0; i < vec1->length; i++) {
    vec1->mat1D->values[i][0] += vec2->mat1D->values[i][0];
  }
}

/**
 * @brief sub Vector vec1 from Vector vec2 (vec1 - vec2)
 *
 * @param vec1
 * @param vec2 (const)
 */
void dv_sub_vector(DoubleVector *vec1, const DoubleVector *vec2) {
  if (vec1->length != vec2->length) {
    perror("vectors are not same length");
  }

  for (size_t i = 0; i < vec1->length; i++) {
    vec1->mat1D->values[i][0] -= vec2->mat1D->values[i][0];
  }
}

/**
 * @brief multiply each element of Vector vec1 with a scalar
 *
 * @param vec
 * @param scalar
 */
void dv_multiply_by_scalar(DoubleVector *vec, const double scalar) {
  for (size_t i = 0; i < vec->length; i++) {
    vec->mat1D->values[i][0] = vec->mat1D->values[i][0] * scalar;
  }
}

/**
 * @brief divied each element of Vector vec1 with a scalar
 *
 * @param vec
 * @param scalar
 */
void dv_divide_by_scalar(DoubleVector *vec, const double scalar) {
  for (size_t i = 0; i < vec->length; i++) {
    vec->mat1D->values[i][0] = vec->mat1D->values[i][0] / scalar;
  }
}

/**
 * @brief add constant to vector
 *
 * @param vec
 * @param scalar
 */
void dv_add_constant(DoubleVector *vec, const double constant) {
  for (size_t i = 0; i < vec->length; i++) {
    vec->mat1D->values[i][0] = vec->mat1D->values[i][0] + constant;
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
  if (vec1->length != vec2->length) {
    return false;
  }

  for (size_t i = 0; i < vec1->length; i++) {
    if (vec1->mat1D->values[i][0] != vec2->mat1D->values[i][0]) {
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
  for (size_t i = 0; i < vec->length; i++) {
    mean += vec->mat1D->values[i][0];
  }

  return (mean / (double)vec->length);
}

/**
 * @brief return min of Vector vec
 *
 * @param vec
 * @return double
 */
double dv_min(DoubleVector *vec) {
  double min = vec->mat1D->values[0][0];
  for (size_t i = 0; i < vec->length; i++) {
    if (min > vec->mat1D->values[i][0]) {
      min = vec->mat1D->values[i][0];
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
  double max = vec->mat1D->values[0][0];
  for (size_t i = 0; i < vec->length; i++) {
    if (max < vec->mat1D->values[i][0]) {
      max = vec->mat1D->values[i][0];
    }
  }
  return max;
}

/**
 * @brief retrun adress of row vector
 *
 * @param mat
 * @param row
 * @return DoubleVector*
 */
double *dm_get_row_as_array(const DoubleMatrix *mat, size_t row) {
  if (row < 0 || row > (mat->rows - 1)) {
    perror("This row does not exist");
  }
  return (mat->values[row]);
}
