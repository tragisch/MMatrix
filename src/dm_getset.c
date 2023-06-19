/**
 * @file dm_set.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dbg.h"
#include "dm.h"
#include "dm_internals.h"
#include "dm_io.h"
#include "dm_math.h"
#include "dm_vector.h"

/*******************************/
/*          Set Value          */
/*******************************/

/**
 * @brief set value of index i, j
 *
 * @param mat
 * @param i,j
 * @param value
 */
void dm_set(DoubleMatrix *mat, size_t i, size_t j, double value) {
  if (i < 0 || i >= mat->rows || j < 0 || j >= mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    exit(EXIT_FAILURE);
  }
  switch (mat->format) {
  case COO:
    dm_set_coo(mat, i, j, value);
    break;
  case CSC:
    dm_set_csc(mat, i, j, value);
    break;
  case DENSE:
    dm_set_dense(mat, i, j, value);
    break;
  case VECTOR:
    dv_set(mat, i, value);
    break;
  }
}

/*******************************/
/*         Set DENSE           */
/*******************************/

// get value from dense matrix:
static void dm_set_dense(DoubleMatrix *mat, size_t i, size_t j,
                         const double value) {
  mat->values[i * mat->cols + j] = value;
  if (value != 0.0) {
    mat->nnz++;
  }
}

/*******************************/
/*         Set COO          */
/*******************************/

static void dm_set_coo(DoubleMatrix *matrix, size_t i, size_t j, double value) {
  // Find the position of the element (i, j) in the matrix
  size_t position = binary_search_coo(matrix, i, j);

  if (position < matrix->nnz && matrix->row_indices[position] == i &&
      matrix->col_indices[position] == j) {
    // Element already exists at position (i, j), update the value
    matrix->values[position] = value;
  } else {
    // Insert new element at the appropriate position
    // and shift the existing elements to make space
    insert_element(matrix, i, j, value, position);
  }
}

size_t binary_search_coo(const DoubleMatrix *matrix, size_t i, size_t j) {
  size_t low = 0;
  size_t high = matrix->nnz;

  while (low < high) {
    size_t mid = (low + high) / 2;

    if (matrix->row_indices[mid] == i && matrix->col_indices[mid] == j) {
      return mid; // Element found at position (i, j)
    }
    if (matrix->row_indices[mid] < i ||
        (matrix->row_indices[mid] == i && matrix->col_indices[mid] < j)) {
      low = mid + 1; // Search in the upper half
    } else {
      high = mid; // Search in the lower half
    }
  }

  return low; // Element not found, return the insertion position
}

void insert_element(DoubleMatrix *matrix, size_t i, size_t j, double value,
                    size_t position) {
  // Increase the capacity if needed
  if (matrix->nnz == matrix->capacity) {
    dm_realloc_coo(matrix, matrix->capacity * 2);
  }

  // Shift the existing elements to make space for the new element
  for (size_t k = matrix->nnz; k > position; k--) {
    matrix->row_indices[k] = matrix->row_indices[k - 1];
    matrix->col_indices[k] = matrix->col_indices[k - 1];
    matrix->values[k] = matrix->values[k - 1];
  }

  // Insert the new element at the appropriate position
  matrix->row_indices[position] = i;
  matrix->col_indices[position] = j;
  matrix->values[position] = value;

  // Increment the count of non-zero elements
  matrix->nnz++;
}

/*******************************/
/*          Set CSC          */
/*******************************/

static void dm_set_csc(DoubleMatrix *mat, size_t i, size_t j, double value) {

  size_t col_start = mat->col_ptrs[j];
  size_t col_end = mat->col_ptrs[j + 1];

  // Check if the element already exists in the column
  for (size_t k = col_start; k < col_end; k++) {
    if (mat->row_indices[k] == i) {
      mat->values[k] = value;
      return;
    }
  }

  // The element doesn't exist, so we need to insert it

  // Check if there is enough capacity to insert a new element
  if (mat->nnz + 1 > mat->capacity) {
    dm_realloc_csc(mat, mat->capacity * 2);
  }

  // Shift the elements after the insertion point to make space
  for (size_t k = mat->nnz; k > col_start; k--) {
    mat->row_indices[k] = mat->row_indices[k - 1];
    mat->values[k] = mat->values[k - 1];
  }

  // Insert the new element
  mat->row_indices[col_start] = i;
  mat->values[col_start] = value;

  // Update the column pointers after the insertion point
  for (size_t k = j + 1; k <= mat->cols; k++) {
    mat->col_ptrs[k]++;
  }

  mat->nnz++;
}

/*******************************/
/*          Get Value          */
/*******************************/

/**
 * @brief get value of index i, j
 *
 * @param mat
 * @param i,j
 * @return double
 */
double dm_get(const DoubleMatrix *mat, size_t i, size_t j) {
  // perror if boundaries are exceeded
  if (i < 0 || i > mat->rows || j < 0 || j > mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    exit(EXIT_FAILURE);
  }
  switch (mat->format) {
  case DENSE:
    return dm_get_dense(mat, i, j);
    break;
  case COO:
    return dm_get_coo(mat, i, j);
    break;
  case CSC:
    return dm_get_csc(mat, i, j);
    break;
  case VECTOR:
    return dv_get(mat, i);
    break;
  }
}

/*******************************/
/*         Get DENSE           */
/*******************************/

// get value from dense matrix:
static double dm_get_dense(const DoubleMatrix *mat, size_t i, size_t j) {
  return mat->values[i * mat->cols + j];
}

/*******************************/
/*         Get CSC             */
/*******************************/

static double dm_get_csc(const DoubleMatrix *mat, size_t i, size_t j) {
  size_t col_start = mat->col_ptrs[j];
  size_t col_end = mat->col_ptrs[j + 1];

  for (size_t k = col_start; k < col_end; k++) {

    if (mat->row_indices[k] == i) {
      return mat->values[k];
    }
  }

  // Element not found, return 0.0
  return 0.0;
}

/*******************************/
/*         Get COO          */
/*******************************/

static double dm_get_coo(const DoubleMatrix *matrix, size_t i, size_t j) {
  if (i >= matrix->rows || j >= matrix->cols) {
    // Invalid position, handle error accordingly
    return 0.0; // Assuming 0.0 represents the default value
  }

  size_t position = binary_search_coo(matrix, i, j);

  if (position < matrix->nnz && matrix->row_indices[position] == i &&
      matrix->col_indices[position] == j) {
    // Element found at position (i, j)
    return matrix->values[position];
  }

  // Element not found at position (i, j), return 0.0 or the default value
  return 0.0; // Assuming 0.0 represents the default value
}

/*******************************/
/*         Get VECTOR          */
/*******************************/

// see dv_vector.c
