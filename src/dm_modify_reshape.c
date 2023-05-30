/**
 * @file dm_modify_reshape.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm.h"
#include "dm_modify.h"

/*******************************/
/*         Reshape Matrix      */
/*******************************/

// Reshape matrix to a new numer of columns and rows without changing its data.
void dm_reshape(DoubleMatrix *mat, size_t new_rows, size_t new_cols) {
  // Check if the new number of rows and columns is valid

  if (new_rows * new_cols != mat->rows * mat->cols) {
    perror("The new number of rows * columns must be equal to the old number "
           "of rows * columns.");
    exit(EXIT_FAILURE);
  }

  switch (mat->format) {
  case DENSE:
    dm_reshape_dense(mat, new_rows, new_cols);
    break;
  case COO:
    dm_reshape_sparse(mat, new_rows, new_cols);
    break;
  case CSR:
    break; // not implemented yet
  case VECTOR:
    break;
  }
}

/*******************************/
/*       Reshape DENSE         */
/*******************************/

static void dm_reshape_dense(DoubleMatrix *matrix, size_t new_rows,
                             size_t new_cols) {

  DoubleMatrix *reshaped_matrix = dm_create_format(new_rows, new_cols, DENSE);
  if (reshaped_matrix == NULL) {
    // Failed to allocate memory for the reshaped matrix
    return;
  }

  // Copy values from the original matrix to the reshaped matrix
  memcpy(reshaped_matrix->values, matrix->values,
         matrix->rows * matrix->cols * sizeof(double));

  // Update the matrix pointer with the reshaped matrix
  matrix->rows = new_rows;
  matrix->cols = new_cols;
  matrix->capacity = new_rows * new_cols;
  free(matrix->values);
  matrix->values = reshaped_matrix->values;

  // Free the reshaped matrix struct (not the values as they were assigned to
  // the original matrix)
  free(reshaped_matrix);
}

/*******************************/
/*       Reshape COO        */
/*******************************/
// TODO: Implement
static void dm_reshape_sparse(DoubleMatrix *matrix, size_t new_rows,
                              size_t new_cols) {}
