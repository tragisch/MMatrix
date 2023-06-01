/**
 * @file dm_realloc.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm.h"

/*******************************/
/*      Realloc COO          */
/*******************************/

void dm_realloc_coo(DoubleMatrix *mat, size_t new_capacity) {
  // check if matrix is already in sparse format:
  if (mat->format == DENSE) {
    printf("Can not resize matrix to sparse format!\n");
    exit(EXIT_FAILURE);
  }

  if (new_capacity <= mat->capacity) {
    printf("Can not resize matrix to smaller capacity!\n");
    return;
  }

  // resize matrix:
  size_t *row_indices = (size_t *)realloc(
      mat->row_indices, (mat->capacity + new_capacity) * sizeof(size_t));
  size_t *col_indices = (size_t *)realloc(
      mat->col_indices, (mat->capacity + new_capacity) * sizeof(size_t));
  double *values = (double *)realloc(
      mat->values, (mat->capacity + new_capacity) * sizeof(double));
  if (row_indices == NULL || col_indices == NULL || values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  mat->capacity += new_capacity;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
}

/*******************************/
/*       Realloc DENSE         */
/*******************************/
