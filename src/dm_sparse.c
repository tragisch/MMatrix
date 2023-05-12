/**
 * @file dm_csr_matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 17-04-2023
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "dbg.h"
#include "dm_io.h"
#include "dm_math.h"
#include "dm_matrix.h"

enum { INIT_CAPACITY = 2U };
enum { RESIZE_FACTOR = 2 };

/*******************************/
/*        CSR Sparse Matrix    */
/*******************************/

DoubleMatrix *dm_create_sparse(size_t rows, size_t cols) {
  if (rows < 1 || cols < 1) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  DoubleMatrix *mat = malloc(sizeof(DoubleMatrix));
  mat->rows = rows;
  mat->cols = cols;
  mat->row_capacity = rows + INIT_CAPACITY;
  mat->col_capacity = rows + INIT_CAPACITY;
  mat->nnz = 0;
  mat->row_pointers = calloc(mat->row_capacity, sizeof(size_t));
  mat->col_indices =
      calloc(max_int(INIT_CAPACITY, (int)mat->nnz), sizeof(size_t));
  mat->format = SPARSE;
  mat->values = calloc(max_int(INIT_CAPACITY, (int)mat->nnz), sizeof(double));
  return mat;
}

/*******************************/
/*        Setter & Getter      */
/*******************************/

// void dm_set_sparse(DoubleMatrix *mat, size_t i, size_t j, double value) {
//   bool found = false;
//   for (size_t k = mat->row_pointers[i]; k < mat->row_pointers[i + 1]; k++) {
//     if (mat->col_indices[k] == j) {
//       if (!dm_is_zero(value)) {
//         mat->values[k] = value;
//       } else { // remove existing non-zero value
//         mat->col_indices[k] = mat->col_indices[mat->nnz - 1];
//         mat->values[k] = mat->values[mat->nnz - 1];
//         mat->row_pointers[i + 1]--;
//         mat->nnz--;
//       }
//       found = true;
//       break;
//     }
//   }
//   if (!found) { // insert new value
//     if (!dm_is_zero(value)) {
//       size_t insert_pos = mat->row_pointers[i + 1];
//       mat->col_indices =
//           realloc(mat->col_indices, (mat->nnz + 1) * sizeof(size_t));
//       mat->values = realloc(mat->values, (mat->nnz + 1) * sizeof(double));
//       for (size_t k = mat->nnz; k > insert_pos; k--) {
//         mat->col_indices[k] = mat->col_indices[k - 1];
//         mat->values[k] = mat->values[k - 1];
//       }
//       mat->col_indices[insert_pos] = j;
//       mat->values[insert_pos] = value;
//       mat->row_pointers[i + 1]++;
//       mat->nnz++;
//     }
//   }
// }

void dm_set_sparse(DoubleMatrix *mat, size_t i, size_t j, double value) {
  // check if indices are valid:
  if (i >= mat->rows || j >= mat->cols) {
    perror("Error: invalid matrix indices.\n");
    return;
  }
  // check if value is zero:
  if (dm_is_zero(value)) {
    dm_remove_zero(mat, i, j);
  } else {
    // check if col_capacity is sufficient:
    dm_realloc_col_ind_val(mat);
    // insert non-zero value:
    setL(mat, i, j, value);
  }
  // print to console: value und i, j
  // printf("value: %f, i: %zu, j: %zu\n", dm_get(mat, i, j), i, j);
}

// static function to remove a zero value from a sparse matrix:
static void dm_remove_zero(DoubleMatrix *mat, size_t i, size_t j) {
  // check if indices are valid:
  if (i >= mat->rows || j >= mat->cols) {
    perror("Error: invalid matrix indices.\n");
    return;
  }
  // remove existing non-zero value:
  for (size_t k = mat->row_pointers[i]; k < mat->row_pointers[i + 1]; k++) {
    if (mat->col_indices[k] == j) {
      mat->col_indices[k] = mat->col_indices[mat->nnz - 1];
      mat->values[k] = mat->values[mat->nnz - 1];
      mat->row_pointers[i + 1]--;
      mat->nnz--;
      break;
    }
  }
}

static void dm_set_non_zero(DoubleMatrix *mat, size_t i, size_t j,
                            double value) {
  size_t row_start = mat->row_pointers[i];
  size_t row_end = mat->row_pointers[i + 1];

  // check if the entry j already exists
  for (size_t k = row_start; k < row_end; k++) {
    if (mat->col_indices[k] == j) {
      mat->values[k] = value;
      return;
    }
  }

  // entry (i,j) does not exist, so add it
  size_t nnz = mat->nnz;

  for (size_t k = nnz; k > row_end; k--) {
    mat->col_indices[k] = mat->col_indices[k - 1];
    mat->values[k] = mat->values[k - 1];
  }
  mat->col_indices[row_end] = j;
  mat->values[row_end] = value;
  mat->nnz++;

  // update nnz counts for all subsequent rows
  for (size_t k = i + 1; k < mat->rows; k++) {
    mat->row_pointers[k]++;
  }
}

static void setL(DoubleMatrix *mat, size_t i, size_t j, double value) {
  if (i >= mat->rows || j >= mat->cols) {
    // Indices out of range, handle the error
    return;
  }

  size_t row_start = mat->row_pointers[i];
  size_t row_end = mat->row_pointers[i + 1];

  // Find the appropriate position to insert the new element
  size_t insert_pos = row_start;
  while (insert_pos < row_end && mat->col_indices[insert_pos] < j) {
    insert_pos++;
  }

  if (insert_pos < row_end && mat->col_indices[insert_pos] == j) {
    // Element already exists, update its value
    mat->values[insert_pos] = value;
  } else {

    // Shift elements to the right to make space for the new element
    for (size_t k = row_end; k > insert_pos; k--) {
      mat->col_indices[k] = mat->col_indices[k - 1];
      mat->values[k] = mat->values[k - 1];
    }

    // Insert the new element at the appropriate position
    mat->col_indices[insert_pos] = j;

    mat->values[insert_pos] = value;

    // Update the row pointers for subsequent rows
    for (size_t k = i + 1; k <= mat->rows; k++) {
      mat->row_pointers[k]++;
    }

    // Update the number of non-zero elements
    mat->nnz++;
  }
}

// static function the reallocs col_indes and values if necessary:
static void dm_realloc_col_ind_val(DoubleMatrix *mat) {
  if (mat->col_capacity < (mat->nnz + INIT_CAPACITY)) {
    size_t *col_ind_tmp = realloc(
        mat->col_indices, (mat->col_capacity + INIT_CAPACITY) * sizeof(size_t));
    double *values_tmp = realloc(
        mat->values, (mat->col_capacity + INIT_CAPACITY) * sizeof(double));
    if (col_ind_tmp == NULL || values_tmp == NULL) {
      perror("Error: memory allocation failed.\n");
      exit(EXIT_FAILURE);
    }
    mat->col_capacity += INIT_CAPACITY;
    mat->col_indices = col_ind_tmp;
    mat->values = values_tmp;
  }
}

double dm_get_sparse(const DoubleMatrix *mat, size_t i, size_t j) {

  size_t row_start = mat->row_pointers[i];
  size_t row_end = mat->row_pointers[i + 1];

  for (size_t k = row_start; k < row_end; k++) {
    if (mat->col_indices[k] == j) {
      return mat->values[k];
    }
  }

  // Element not found, return 0.0
  return 0.0;
}

/*******************************/
/*        Convert               */
/*******************************/

// convert matrix to sparse format:
void dm_convert_to_sparse(DoubleMatrix *mat) {
  // check if matrix is already in sparse format:
  if (mat->format == SPARSE) {
    // nothing to do
  } else {

    // create new sparse matrix:
    DoubleMatrix *new_mat = dm_create_sparse(mat->rows, mat->cols);
    // copy values:
    for (size_t i = 0; i < mat->rows; i++) {
      for (size_t j = 0; j < mat->cols; j++) {
        double value = dm_get(mat, i, j);
        if (dm_is_zero(value) == false) {
          dm_set(new_mat, i, j, value);
        }
      }
    }

    // free memory allocated for original matrix:
    free(mat->values);
    free(mat->row_pointers);
    free(mat->col_indices);

    // update pointer to new sparse matrix:
    mat->values = new_mat->values;
    mat->row_pointers = new_mat->row_pointers;
    mat->col_indices = new_mat->col_indices;
    mat->nnz = new_mat->nnz;
    mat->row_capacity = new_mat->row_capacity;
    mat->col_capacity = new_mat->col_capacity;
    mat->format = new_mat->format;

    // set new dimensions:
    mat->rows = new_mat->rows;
    mat->cols = new_mat->cols;

    // free memory allocated for new matrix:
    free(new_mat);
  }
}

/*******************************/
/*        Resize               */
/*******************************/

// Helper function to compute the new row pointers and nnz
static void compute_new_row_pointers_and_nnz(DoubleMatrix *mat, size_t new_rows,
                                             size_t *new_nnz,
                                             size_t *new_row_capacity) {
  if (new_rows == mat->rows) {
    *new_nnz = mat->nnz;
    *new_row_capacity = mat->row_capacity;
    return;
  }

  *new_nnz = mat->nnz;
  *new_row_capacity = mat->row_capacity;

  if (new_rows > mat->rows) {
    *new_row_capacity = new_rows + 1;
    *new_nnz = mat->row_pointers[mat->rows];
  } else {
    for (size_t i = new_rows; i < mat->rows; i++) {
      *new_nnz -= mat->row_pointers[i];
    }
  }
}

// Helper function to compute the new column capacity
static size_t compute_new_col_capacity(DoubleMatrix *mat, size_t new_cols) {
  if (new_cols <= mat->col_capacity) {
    return mat->col_capacity;
  }

  size_t new_col_capacity = (size_t)(mat->col_capacity * (1 + INIT_CAPACITY));
  if (new_col_capacity < new_cols) {
    new_col_capacity = new_cols;
  }

  return new_col_capacity;
}

// Helper function to resize the column indices and values arrays
static void resize_col_arrays(DoubleMatrix *mat, size_t new_col_capacity) {
  size_t *new_col_indices =
      (size_t *)realloc(mat->col_indices, new_col_capacity * sizeof(size_t));
  double *new_values =
      (double *)realloc(mat->values, new_col_capacity * sizeof(double));
  if (new_col_indices == NULL || new_values == NULL) {
    fprintf(stderr,
            "Error: Failed to allocate memory for column indices or values\n");
    exit(EXIT_FAILURE);
  }

  mat->col_indices = new_col_indices;
  mat->values = new_values;
  mat->col_capacity = new_col_capacity;
}

// Main resize function
void dm_resize_sparse(DoubleMatrix *mat, size_t new_rows, size_t new_cols) {
  if (new_rows == mat->rows && new_cols == mat->cols) {
    return;
  }
  // Compute the new row pointers and nnz
  size_t new_nnz = 0;
  size_t new_row_capacity = 0;
  compute_new_row_pointers_and_nnz(mat, new_rows, &new_nnz, &new_row_capacity);

  // Compute the new column capacity
  size_t new_col_capacity = compute_new_col_capacity(mat, new_cols);
  resize_col_arrays(mat, new_col_capacity);

  // Resize the row pointers
  size_t *new_row_pointers =
      (size_t *)realloc(mat->row_pointers, new_row_capacity * sizeof(size_t));
  if (new_row_pointers == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for row pointers\n");
    exit(EXIT_FAILURE);
  }
  mat->row_pointers = new_row_pointers;

  // Update the row pointers
  for (size_t i = mat->rows; i <= new_rows; i++) {
    mat->row_pointers[i] = new_nnz;
  }
  // Update the dimensions
  mat->rows = new_rows;
  mat->cols = new_cols;
  mat->nnz = new_nnz;
  mat->row_capacity = new_row_capacity;
}
