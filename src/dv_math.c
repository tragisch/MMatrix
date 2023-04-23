/**
 * @file dv_math.c
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
#include "dv_vector.h"

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





