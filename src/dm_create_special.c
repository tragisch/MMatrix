/**
 * @file sp_special.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 17-04-2023
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "dm.h"
#include "dm_io.h"
#include "dm_math.h"
#include "dm_modify.h"

/*******************************/
/*        Special Matrix        */
/*******************************/

/**
 * @brief Create a Identity object
 *
 * @param rows
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_identity(size_t rows) {
  DoubleMatrix *mat = dm_create_format(rows, rows, default_matrix_format);
  for (size_t i = 0; i < rows; i++) {
    dm_set(mat, i, i, 1.0);
  }

  return mat;
}

/**
 * @brief create a Double Matrix with random elements in range [0,1]
 *
 * @param rows
 * @param cols
 * @param density
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_rand(size_t rows, size_t cols, double density) {
  DoubleMatrix *mat = dm_create_format(rows, cols, default_matrix_format);

  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      if (randomDouble() <= density) {
        double value = randomDouble();
        dm_set(mat, i, j, value);
      }
    }
  }
  return mat;
}

/**
 * @brief create a Double Matrix with random elements in range [min,max]
 *
 * @param rows
 * @param cols
 * @param min
 * @param max
 * @param density
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_rand_between(size_t rows, size_t cols, size_t min,
                                     size_t max, double density) {
  DoubleMatrix *mat = dm_create_format(rows, cols, default_matrix_format);

  for (int i = 0; i < mat->rows; i++) {
    for (int j = 0; j < mat->cols; j++) {
      if (randomDouble() <= density) {
        double value = randomDouble_betweenBounds(min, max);
        dm_set(mat, i, j, value);
      }
    }
  }
  return mat;
}

/**
 * @brief Set the Array To Matrix object
 *
 * @param rows
 * @param cols
 * @param array
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_from_array(size_t rows, size_t cols,
                                   double array[rows][cols]) {
  DoubleMatrix *mat = dm_create_format(rows, cols, default_matrix_format);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, array[i][j]);
    }
  }

  return mat;
}

/**
 * @brief Create a Diagonal object
 *
 * @param rows
 * @param cols
 * @param array
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_diagonal(size_t rows, size_t cols, double array[rows]) {
  DoubleMatrix *mat = dm_create_format(rows, cols, default_matrix_format);

  dm_print(mat);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      if (i == j) {
        dm_set(mat, i, j, array[i]);
      } else if ((i != j) && (mat->format == DENSE)) {
        dm_set(mat, i, j, 0.0);
      }
    }
  }

  // remove small entries
  if ((mat->format == COO) || (mat->format == CSC)) {
    dm_drop_small_entries(mat);
  }

  return mat;
}
