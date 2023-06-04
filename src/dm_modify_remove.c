/**
 * @file dm_modify_remove.c
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
#include "dm_io.h"
#include "dm_modify.h"
#include <string.h>

/*******************************/
/*       Remove  entry         */
/*******************************/

/**
 * @brief Remove a non-zero element from a sparse matrix
 *
 * @param mat (COO or CSC format)
 * @param i
 * @param j
 */
void dm_remove_entry(DoubleMatrix *mat, size_t i, size_t j) {
  switch (mat->format) {
  case COO:
    dm_remove_entry_coo(mat, i, j);
    break;
  case CSC:
    dm_remove_entry_csc(mat, i, j);
    break;
  default:
    perror("This function is only implemented for sparse matrices");
  }
}

// remove nnz value at index i,j of sparse matrix in COO format:
// TODO: if removed element is last of row, col --> remove row, col
// Due to performance reasons, there is an overall check during odering COO

static void dm_remove_entry_coo(DoubleMatrix *mat, size_t i, size_t j) {
  for (int k = 0; k < mat->nnz; k++) {
    if ((mat->row_indices[k] == i) && (mat->col_indices[k] == j)) {
      // remove element if found
      mat->nnz--;
      for (int l = k; l < mat->nnz; l++) {
        mat->row_indices[l] = mat->row_indices[l + 1];
        mat->col_indices[l] = mat->col_indices[l + 1];
        mat->values[l] = mat->values[l + 1];
      }
      return;
    }
  }
}

static void dm_remove_entry_csc(DoubleMatrix *mat, size_t i, size_t j) {
  size_t idx = 0;

  // Search for the row i in column j
  for (size_t k = mat->col_ptrs[j]; k < mat->col_ptrs[j + 1]; k++) {
    if (mat->row_indices[k] == i) {
      idx = k;
      break;
    }
  }

  // Shift entries in values[] and rowIndices[] to remove the entry at idx
  for (size_t k = idx; k < mat->col_ptrs[mat->cols] - 1; k++) {
    mat->values[k] = mat->values[k + 1];
    mat->row_indices[k] = mat->row_indices[k + 1];
  }

  // Update colPointers[] for all columns j' > j
  for (size_t k = j + 1; k <= mat->cols; k++) {
    mat->col_ptrs[k]--;
  }

  mat->nnz--;

  double *new_values = realloc(mat->values, mat->nnz * sizeof(double));
  size_t *new_row_indices =
      realloc(mat->row_indices, mat->nnz * sizeof(size_t));

  if (new_values == NULL || new_row_indices == NULL) {
    perror("Could not reallocate memory");
    exit(EXIT_FAILURE);
  }

  mat->values = new_values;
  mat->row_indices = new_row_indices;
}

/*******************************/
/*         Remove Column       */
/*******************************/

void dm_remove_column(DoubleMatrix *mat, size_t column_idx) {
  if (column_idx < 0 || column_idx > (mat->cols - 1)) {
    perror("This column does not exist");
  }
  switch (mat->format) {
  case DENSE:
    dm_remove_column_dense(mat, column_idx);
    break;
  case COO:
    dm_remove_column_coo(mat, column_idx);
    break;
  case CSC:
    dm_remove_column_csc(mat, column_idx);
    break;
  case VECTOR:
    break;
  }
}

static void dm_remove_column_coo(DoubleMatrix *mat, size_t column_idx) {

  // Step 1: Identify the indices to remove
  size_t *indices_to_remove = malloc(mat->nnz * sizeof(size_t));
  size_t num_indices_to_remove = 0;
  for (size_t i = 0; i < mat->nnz; i++) {
    if (mat->col_indices[i] == column_idx) {
      indices_to_remove[num_indices_to_remove] = i;
      num_indices_to_remove++;
    }
  }

  // Step 2: Remove the elements
  for (size_t i = num_indices_to_remove; i > 0; i--) {
    size_t index_to_remove = indices_to_remove[i - 1];
    memmove(&mat->values[index_to_remove], &mat->values[index_to_remove + 1],
            (mat->nnz - index_to_remove - 1) * sizeof(double));
    memmove(&mat->row_indices[index_to_remove],
            &mat->row_indices[index_to_remove + 1],
            (mat->nnz - index_to_remove - 1) * sizeof(size_t));
    memmove(&mat->col_indices[index_to_remove],
            &mat->col_indices[index_to_remove + 1],
            (mat->nnz - index_to_remove - 1) * sizeof(size_t));
  }

  // Step 3: Resize the arrays
  if (mat->nnz >= num_indices_to_remove) {
    mat->nnz = (mat->nnz - num_indices_to_remove);
  } else {
    mat->nnz = 1;
  }
  double *new_values = realloc(mat->values, mat->nnz * sizeof(double));
  size_t *new_row_indices =
      realloc(mat->row_indices, mat->nnz * sizeof(size_t));
  size_t *new_col_indices =
      realloc(mat->col_indices, mat->nnz * sizeof(size_t));

  if (new_values == NULL || new_row_indices == NULL ||
      new_col_indices == NULL) {
    perror("Could not reallocate memory");
    exit(EXIT_FAILURE);
  }

  mat->values = new_values;
  mat->row_indices = new_row_indices;
  mat->col_indices = new_col_indices;

  free(indices_to_remove);

  // resize the matrix:
  dm_resize(mat, mat->rows, mat->cols - 1);
}

static void dm_remove_column_csc(DoubleMatrix *mat, size_t col_idx) {
  // nnz of column to be removed
  size_t nnz_col_idx = (mat->col_ptrs[col_idx + 1] - mat->col_ptrs[col_idx]);

  if (col_idx == mat->cols - 1) {
    mat->cols--;
    mat->col_ptrs[mat->cols] = mat->nnz - nnz_col_idx;
    mat->nnz -= nnz_col_idx;
    // mat->cols--;
  } else {

    size_t bytes_after_remove_col_idx = (mat->cols - 1) * sizeof(size_t);
    size_t remove_start = mat->col_ptrs[col_idx] - nnz_col_idx;
    size_t remove_end = mat->col_ptrs[col_idx + 1] - nnz_col_idx;
    size_t bytes_after_remove_end = (mat->nnz - remove_end) * sizeof(size_t);

    memmove(&mat->col_ptrs[col_idx], &mat->col_ptrs[col_idx + 1],
            bytes_after_remove_col_idx);
    memmove(&mat->row_indices[remove_start], &mat->row_indices[remove_end],
            bytes_after_remove_end);
    bytes_after_remove_end = (mat->nnz - remove_end) * sizeof(double);
    memmove(&mat->values[remove_start], &mat->values[remove_end],
            bytes_after_remove_end);

    mat->nnz -= nnz_col_idx;
    mat->cols--;
    for (size_t i = col_idx; i < mat->cols; i++) {
      mat->col_ptrs[i] -= nnz_col_idx;
    }
    mat->col_ptrs[mat->cols] = mat->nnz;
  }

  // Optionally, reallocate memory to free unused space
  size_t *new_col_ptrs =
      realloc(mat->col_ptrs, (mat->cols + 1) * sizeof(size_t));
  size_t *new_row_indices =
      realloc(mat->row_indices, mat->nnz * sizeof(size_t));
  double *new_values = realloc(mat->values, mat->nnz * sizeof(double));

  if ((new_col_ptrs != NULL) || (new_row_indices != NULL) ||
      (new_values != NULL)) {
    mat->col_ptrs = new_col_ptrs;
    mat->row_indices = new_row_indices;
    mat->values = new_values;
  }
}

static void dm_remove_column_dense(DoubleMatrix *mat, size_t column_idx) {
  // shift all columns to the left:
  for (size_t i = column_idx; i < mat->cols - 1; i++) {
    for (size_t j = 0; j < mat->rows; j++) {
      dm_set(mat, j, i, dm_get(mat, j, i + 1));
    }
  }

  // resize the matrix:
  dm_resize(mat, mat->rows, mat->cols - 1);
}

/*******************************/
/*         Remove Row          */
/*******************************/

void dm_remove_row(DoubleMatrix *mat, size_t row_idx) {
  if (row_idx < 0 || row_idx > (mat->rows - 1)) {
    perror("This row does not exist");
  }
  switch (mat->format) {
  case DENSE:
    dm_remove_row_dense(mat, row_idx);
    break;
  case COO:
    dm_remove_row_coo(mat, row_idx);
    break;
  case CSC:
    dm_remove_row_csc(mat, row_idx);
    break;
  case VECTOR:
    break;
  }
}

// static void dm_remove_row_coo(DoubleMatrix *mat, size_t row_idx) {
//   size_t *temp_row_index = calloc(mat->nnz, sizeof(size_t));
//   size_t *temp_col_index = calloc(mat->nnz, sizeof(size_t));
//   double *temp_values = calloc(mat->nnz, sizeof(double));
//   size_t new_nnz = 0;

//   // copy all values except the ones in the row to be removed:
//   int k = 0;
//   for (size_t i = 0; i < mat->nnz; i++) {
//     if (mat->row_indices[i] != row_idx) {
//       temp_row_index[k] = mat->row_indices[i];
//       temp_col_index[k] = mat->col_indices[i];
//       temp_values[k] = mat->values[i];
//       new_nnz++;
//       k++;
//     }
//   }

//   // shift all rows to the top:
//   for (size_t i = 0; i < mat->nnz; i++) {
//     if (temp_row_index[i] >= row_idx) {
//       temp_row_index[i]--;
//     }
//   }

//   // copy the values back:
//   free(mat->row_indices);
//   free(mat->col_indices);
//   free(mat->values);
//   mat->row_indices = temp_row_index;
//   mat->col_indices = temp_col_index;
//   mat->values = temp_values;
//   mat->rows--;
//   mat->nnz = new_nnz;
// }

static void dm_remove_row_coo(DoubleMatrix *mat, size_t row_idx) {

  // Find the start and end indices of the row to be removed
  size_t start_idx = 0;
  while (start_idx < mat->nnz && mat->row_indices[start_idx] < row_idx) {
    start_idx++;
  }
  size_t end_idx = start_idx;
  while (end_idx < mat->nnz && mat->row_indices[end_idx] == row_idx) {
    end_idx++;
  }

  // Calculate the number of non-zero elements in the row to be removed
  size_t nnz_removed = end_idx - start_idx;

  // Shift the remaining elements up
  memmove(&mat->row_indices[start_idx], &mat->row_indices[end_idx],
          (mat->nnz - end_idx) * sizeof(size_t));
  memmove(&mat->col_indices[start_idx], &mat->col_indices[end_idx],
          (mat->nnz - end_idx) * sizeof(size_t));
  memmove(&mat->values[start_idx], &mat->values[end_idx],
          (mat->nnz - end_idx) * sizeof(double));

  // Update rows and nnz
  mat->rows--;
  mat->nnz -= nnz_removed;

  // Decrement the row indices of all elements after the removed row
  for (size_t i = start_idx; i < mat->nnz; i++) {
    mat->row_indices[i]--;
  }

  // Optionally, reallocate memory to free unused space
  mat->row_indices = realloc(mat->row_indices, mat->nnz * sizeof(size_t));
  mat->col_indices = realloc(mat->col_indices, mat->nnz * sizeof(size_t));
  mat->values = realloc(mat->values, mat->nnz * sizeof(double));
}

static void dm_remove_row_dense(DoubleMatrix *mat, size_t row_idx) {
  // shift all rows to the top:
  for (size_t i = row_idx; i < mat->rows - 1; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      dm_set(mat, i, j, dm_get(mat, i + 1, j));
    }
  }

  // resize the matrix:
  dm_resize(mat, mat->rows - 1, mat->cols);
}

/*******************************/
/*     Remove Column CSC       */
/*******************************/

static void dm_remove_row_csc(DoubleMatrix *mat, size_t row_idx) {
  // Check for invalid row index
  if (row_idx >= mat->rows) {
    fprintf(stderr, "Invalid row index: %zu\n", row_idx);
    return;
  }

  // Check for empty matrix
  if (mat->rows == 0) {
    fprintf(stderr, "Cannot remove row from empty matrix\n");
    return;
  }

  size_t nnz_removed = 0;

  // Iterate over each column
  for (size_t col_idx = 0; col_idx < mat->cols; col_idx++) {
    size_t col_start = mat->col_ptrs[col_idx];
    size_t col_end = mat->col_ptrs[col_idx + 1];

    // Iterate over each element in the column
    for (size_t i = col_start; i < col_end; i++) {
      if (mat->row_indices[i] == row_idx) {
        // This element belongs to the row to be removed
        nnz_removed++;

        // Shift the remaining elements up
        memmove(&mat->row_indices[i], &mat->row_indices[i + 1],
                (mat->nnz - i - 1) * sizeof(size_t));
        memmove(&mat->values[i], &mat->values[i + 1],
                (mat->nnz - i - 1) * sizeof(double));

        // Update the end of the column
        col_end--;
        i--;
      } else if (mat->row_indices[i] > row_idx) {
        // Decrement the row index
        mat->row_indices[i]--;
      }
    }

    // Update the column pointer
    mat->col_ptrs[col_idx + 1] -= nnz_removed;
  }

  // Update rows and nnz
  mat->rows--;
  mat->nnz -= nnz_removed;

  // Optionally, reallocate memory to free unused space
  mat->row_indices = realloc(mat->row_indices, mat->nnz * sizeof(size_t));
  mat->values = realloc(mat->values, mat->nnz * sizeof(double));
}
