/**
 * @file dm_matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm_matrix.h"
#include "dbg.h"
#include "dm_math.h"

// #define NDEBUG
enum { INIT_CAPACITY = 2U };

/*******************************/
/*        Double Matrix        */
/*******************************/

/**
 * @brief create a Double Matrix Object
 *
 * @param rows
 * @param cols
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create(size_t rows, size_t cols) {
  return dm_create_sparse(rows, cols);
}

// create sparse matrix with given format:
DoubleMatrix *dm_create_format(size_t rows, size_t cols, matrix_format format) {
  DoubleMatrix *mat = NULL;

  switch (format) {
  case SPARSE:
    mat = dm_create_sparse(rows, cols);
    break;
  case DENSE:
    mat = dm_create_dense(rows, cols);
    break;
  case VECTOR:
    mat = dv_create(rows);
    break;
  default:
    perror("Error: invalid matrix format.\n");
    return NULL;
  }

  return mat;
}

/**
 * @brief create a Double Matrix Object
 *
 * @param rows
 * @param cols
 * @param density
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_rand(size_t rows, size_t cols, double density) {
  DoubleMatrix *mat = dm_create(rows, cols);

  // Loop over each element in the matrix
  for (size_t i = 0; i < rows; i++) {
    // size_t nnz_start = mat->row_pointers[i];
    for (size_t j = 0; j < cols; j++) {
      if (randomDouble() <= density) {
        double value = randomDouble();
        if (!dm_is_zero(value)) {
          // Resize col_indices and values arrays if needed
          if (mat->nnz >= (mat->col_capacity + INIT_CAPACITY)) {

            mat->col_capacity += INIT_CAPACITY;
            size_t *tmp_col_indices = (size_t *)realloc(
                mat->col_indices, mat->col_capacity * sizeof(size_t));
            double *tmp_values = (double *)realloc(
                mat->values, mat->col_capacity * sizeof(double));
            if ((tmp_col_indices == NULL) || (tmp_values == NULL)) {
              perror("Error: could not allocate memory for col_indices.\n");
              exit(EXIT_FAILURE);
            }

            mat->col_indices = tmp_col_indices;
            mat->values = tmp_values;
          }
          dm_set(mat, i, j, value);
        }
      }
    }
    // Update the row pointer for the next row
    mat->row_pointers[i + 1] = mat->nnz;
  }

  return mat;
}

/**
 * @brief Create a Identity object
 *
 * @param rows
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_create_identity(size_t rows) {
  DoubleMatrix *mat = dm_create(rows, rows);
  for (size_t i = 0; i < rows; i++) {
    dm_set(mat, i, i, 1.0);
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
  DoubleMatrix *mat = dm_create_dense(rows, cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, array[i][j]);
    }
  }

  return mat;
}

/**
 * @brief return copy of matrix
 *
 * @param m
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_clone(DoubleMatrix *mat) {
  DoubleMatrix *copy = dm_create(mat->rows, mat->cols);
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(copy, i, j, dm_get(mat, i, j));
    }
  }
  return copy;
}

bool dm_is_vector(DoubleMatrix *mat) {
  if ((mat->rows == 1 || mat->cols == 1) && (mat->rows != mat->cols)) {
    return true; // Matrix is a vector
  }
  return false; // Matrix is not a vector
}

/**
 * @brief Resize the matrix
 *
 * @param rows
 * @param cols
 * @param mat
 */

void dm_resize(DoubleMatrix *mat, size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    perror("Matrix dimensions must be greater than 0");
    return;
  }
  switch (mat->format) {
  case DENSE:
    dm_resize_dense(mat, rows, cols);
    break;
  case SPARSE:
    dm_resize_sparse(mat, rows, cols);
    break;
  case VECTOR:
    dm_resize_dense(mat, rows, 1);
  default:
    break;
  }
}

void dm_convert(DoubleMatrix *mat, matrix_format format) {
  if (mat->format == format) {
    return;
  }
  switch (format) {
  case DENSE:
    dm_convert_to_dense(mat);

    break;
  case SPARSE:
    dm_convert_to_sparse(mat);
    break;

  case VECTOR:
    break;
  }
}

/**
 * @brief push (add) a column vector to  matrix
 *
 * @param mat
 * @param col_vec
 */
void dm_push_column(DoubleMatrix *mat, DoubleVector *col_vec) {
  if (mat->rows != col_vec->rows) {
    perror("Error: Length of vector does not fit to number or matrix rows");
  } else {
    // resize the matrix:
    dm_resize(mat, mat->rows, mat->cols + 1);
    for (size_t i = 0; i < mat->rows; i++) {
      dm_set(mat, i, mat->cols - 1, dv_get(col_vec, i));
    }
  }
}

/**
 * @brief push (add) a row vector to matrix
 *
 * @param mat
 * @param row_vec
 */
void dm_push_row(DoubleMatrix *mat, DoubleVector *row_vec) {
  if (row_vec->rows != mat->cols) {
    perror("Error: length of vector does not fit to number or matrix columns");

  } else {
    // resize the matrix:
    dm_resize(mat, mat->rows + 1, mat->cols);
    for (size_t i = 0; i < mat->cols; i++) {
      dm_set(mat, mat->rows - 1, i, dv_get(row_vec, i));
    }
  }
}

/**
 * @brief get value of index i, j
 *
 * @param mat
 * @param i,j
 * @return double
 */
double dm_get(const DoubleMatrix *mat, size_t i, size_t j) {
  // perror if boundaries are exceeded
  if (i >= mat->rows || j >= mat->cols) {
    perror("Error: index out of bounds.\n");
  }
  switch (mat->format) {
  case DENSE:
    return dm_get_dense(mat, i, j);
    break;
  case SPARSE:
    return dm_get_sparse(mat, i, j);
    break;
  case VECTOR:
    return dv_get(mat, i);
    break;
  }
}

/**
 * @brief set value of index i, j
 *
 * @param mat
 * @param i,j
 * @param value
 */
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value) {
  switch (mat->format) {
  case SPARSE:
    dm_set_sparse(mat, i, j, value);
    break;
  case DENSE:
    dm_set_dense(mat, i, j, value);
    break;
  case VECTOR:
    dv_set(mat, i, value);
    break;
  }
}

// free sparse matrix
void dm_destroy(DoubleMatrix *mat) {
  free(mat->col_indices);
  free(mat->values);
  free(mat->row_pointers);
  free(mat);
  mat = NULL;
}

/**
 * @brief get sub matrix of matrix
 *
 * @param mat
 * @param row_start
 * @param row_end
 * @param col_start
 * @param col_end
 * @return DoubleMatrix*
 */
DoubleMatrix *dm_get_sub_matrix(DoubleMatrix *mat, size_t row_start,
                                size_t row_end, size_t col_start,
                                size_t col_end) {
  if (row_start < 0 || row_start > mat->rows || row_end < 0 ||
      row_end > mat->rows || col_start < 0 || col_start > mat->cols ||
      col_end < 0 || col_end > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    exit(EXIT_FAILURE);
  }
  if (row_start > row_end || col_start > col_end) {
    perror("Error: matrix index out of bounds.\n");
    exit(EXIT_FAILURE);
  }
  DoubleMatrix *sub_mat =
      dm_create(row_end - row_start + 1, col_end - col_start + 1);
  for (size_t i = row_start; i <= row_end; i++) {
    for (size_t j = col_start; j <= col_end; j++) {
      dm_set(sub_mat, i - row_start, j - col_start, dm_get(mat, i, j));
    }
  }
  return sub_mat;
}
