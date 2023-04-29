/**
 * @file sp_matrix.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 17-04-2023
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "dbg.h"
#include "dm_math.h"
#include "dm_matrix.h"

/*******************************/
/*        Sparse Matrix        */
/*******************************/

SparseMatrix *sp_create(size_t rows, size_t cols) {
  // check rows and cols
  if (rows < 1 || cols < 1) {
    perror("Invalid matrix dimensions: rows and cols must be > 0");
    return NULL;
  }
  SparseMatrix *sp_matrix = malloc(sizeof(SparseMatrix));
  sp_matrix->rows = rows;
  sp_matrix->cols = cols;
  sp_matrix->nnz = 0;
  sp_matrix->row_indices = malloc(0);
  sp_matrix->col_indices = malloc(0);
  sp_matrix->values = malloc(0);
  sp_matrix->format = COO;
  return sp_matrix;
}

SparseMatrix *sp_create_rand(size_t rows, size_t cols, double density) {
  SparseMatrix *mat = sp_create(rows, cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (randomDouble() < density) {
        double value = randomDouble();
        mat->values = realloc(mat->values, (mat->nnz + 1) * sizeof(double));
        mat->row_indices =
            realloc(mat->row_indices, (mat->nnz + 1) * sizeof(size_t));
        mat->col_indices =
            realloc(mat->col_indices, (mat->nnz + 1) * sizeof(size_t));
        mat->values[mat->nnz] = value;
        mat->row_indices[mat->nnz] = i;
        mat->col_indices[mat->nnz] = j;
        mat->nnz++;
      }
    }
  }

  return mat;
}

// create sparse matrix with given format:
SparseMatrix *sp_create_format(size_t rows, size_t cols, matrix_format format) {
  SparseMatrix *mat = (SparseMatrix *)malloc(sizeof(SparseMatrix));

  mat->rows = rows;
  mat->cols = cols;
  mat->nnz = 0;
  mat->format = format;

  switch (format) {
  case COO:
    mat->row_indices = (size_t *)malloc(0);
    mat->col_indices = (size_t *)malloc(0);
    mat->values = NULL;
    break;
  case CSR:
    mat->row_indices = (size_t *)malloc((rows + 1) * sizeof(size_t));
    mat->col_indices = (size_t *)malloc(0);
    mat->values = (double *)malloc(0);
    break;
  case CSC:
    mat->row_indices = (size_t *)malloc(0);
    mat->col_indices = (size_t *)malloc((cols + 1) * sizeof(size_t));
    mat->values = (double *)malloc(0);
    break;
  case DENSE:
    mat->row_indices = NULL;
    mat->col_indices = NULL;
    mat->values = (double *)malloc((mat->rows * mat->cols) * sizeof(double));
    break;
  }

  return mat;
}

// setup directly COO format for sparse matrix:
SparseMatrix *sp_create_from_array(size_t rows, size_t cols, size_t nnz,
                                   size_t *row_indices, size_t *col_indices,
                                   double *values) {
  SparseMatrix *sp_matrix = sp_create(rows, cols);
  sp_matrix->nnz = nnz;
  sp_matrix->col_indices = col_indices;
  sp_matrix->values = values;
  sp_matrix->row_indices = row_indices;
  sp_matrix->format = COO;
  return sp_matrix;
}

// free sparse matrix
void sp_destroy(SparseMatrix *sp_matrix) {
  if (sp_matrix == NULL) {
    return;
  }
  if (sp_matrix->col_indices != NULL) {
    free(sp_matrix->col_indices);
    sp_matrix->col_indices = NULL;
  }
  if (sp_matrix->values != NULL) {
    free(sp_matrix->values);
    sp_matrix->values = NULL;
  }
  if (sp_matrix->row_indices != NULL) {
    free(sp_matrix->row_indices);
    sp_matrix->row_indices = NULL;
  }
  free(sp_matrix);
}

/*******************************/
/*       GETTER SETTER         */
/*******************************/

double sp_get(const SparseMatrix *mat, size_t i, size_t j) {
  // perror if boundaries are exceeded
  if (i >= mat->rows || j >= mat->cols) {
    perror("Error: index out of bounds.\n");
    return 0.0;
  }
  if (mat->format == DENSE) {
    return dm_get(mat, i, j);
  }

  // search for the element with row i and column j
  for (int k = 0; k < mat->nnz; k++) {
    if (mat->row_indices[k] == i && mat->col_indices[k] == j) {
      return mat->values[k];
    }
  }

  // if not found, return 0.0
  return 0.0;
}

void sp_set(SparseMatrix *mat, size_t i, size_t j, double value) {
  // perror if boundaries are exceeded
  if (i >= mat->rows || j >= mat->cols) {
    perror("Error: index out of bounds.\n");
    return;
  }
  switch (mat->format) {
  case COO:
    sp_set_coo(mat, i, j, value);
    break;
  case CSR:
    sp_set_csr(mat, i, j, value);
    break;
  case CSC:
    sp_set_csc(mat, i, j, value);
    break;
  case DENSE:
    dm_set(mat, i, j, value);
    break;
  }
}

static void sp_set_coo(SparseMatrix *mat, size_t i, size_t j, double value) {
  for (int k = 0; k < mat->nnz; k++) {
    if (mat->row_indices[k] == i && mat->col_indices[k] == j) {
      // update the value if element is found
      mat->values[k] = value;
      return;
    }
  }
}

static void sp_set_csr(SparseMatrix *mat, size_t i, size_t j, double value) {
  // search for the element with row i and column j
  for (int k = mat->row_indices[i]; k < mat->row_indices[i + 1]; k++) {
    if (mat->col_indices[k] == j) {
      // update the value if element is found
      mat->values[k] = value;
      return;
    }
  }
}

static void sp_set_csc(SparseMatrix *mat, size_t i, size_t j, double value) {
  // search for the element with row i and column j
  for (int k = mat->col_indices[j]; k < mat->col_indices[j + 1]; k++) {
    if (mat->row_indices[k] == i) {
      // update the value if element is found
      mat->values[k] = value;
      return;
    }
  }
}

/*
 * @brief resize sparse matrix
 *
 * @param mat
 */
void sp_resize(SparseMatrix *mat, size_t new_rows, size_t new_cols) {
  if (mat == NULL) {
    perror("Error: null pointer passed to sp_resize.\n");
    return;
  }
  if (new_rows < mat->rows || new_cols < mat->cols) {
    perror("Error: cannot shrink matrix.\n");
    return;
  }

  mat->values = realloc(mat->values, sizeof(double) * (mat->nnz));
  mat->row_indices = realloc(mat->row_indices, sizeof(int) * (mat->nnz));
  mat->col_indices = realloc(mat->col_indices, sizeof(int) * (mat->nnz));

  for (int i = mat->rows; i < new_rows; i++) {
    for (int j = 0; j < new_cols; j++) {
      mat->values = realloc(mat->values, sizeof(double) * (mat->nnz + 1));
      mat->row_indices =
          realloc(mat->row_indices, sizeof(int) * (mat->nnz + 1));
      mat->col_indices =
          realloc(mat->col_indices, sizeof(int) * (mat->nnz + 1));
      mat->values[mat->nnz] = 0.0;
      mat->row_indices[mat->nnz] = i;
      mat->col_indices[mat->nnz] = j;
      mat->nnz++;
    }
  }

  mat->rows = new_rows;
  mat->cols = new_cols;
}

// return true if all fields of SparseMatrix a inialized correctly
bool sp_is_valid(const SparseMatrix *a) {
  if (a == NULL) {
    return false;
  }
  if (a->rows == 0 || a->cols == 0) {
    return false;
  }
  if (a->nnz == 0) {
    return false;
  }
  if (a->row_indices == NULL) {
    return false;
  }
  if (a->col_indices == NULL) {
    return false;
  }
  if (a->values == NULL) {
    return false;
  }
  return true;
}

/*******************************/
/*       CONVERT               */
/*******************************/

// convert SparseMatrix to CSC format:
void sp_convert_to_csc(SparseMatrix *mat) {
  if (mat->format == CSC) {
    // Already in CSC format, nothing to do
    return;
  } else if (mat->format == COO || mat->format == CSR) {
    // Allocate memory for new arrays
    size_t *new_row_indices = calloc(mat->cols + 1, sizeof(size_t));
    size_t *new_col_indices = calloc(mat->nnz, sizeof(size_t));
    double *new_values = calloc(mat->nnz, sizeof(double));

    // Compute number of non-zero elements in each column
    for (size_t i = 0; i < mat->nnz; i++) {
      new_row_indices[mat->col_indices[i] + 1]++;
    }
    for (size_t j = 1; j <= mat->cols; j++) {
      new_row_indices[j] += new_row_indices[j - 1];
    }

    // Copy values to new arrays
    for (size_t i = 0; i < mat->rows; i++) {
      for (size_t k = mat->row_indices[i]; k < mat->row_indices[i + 1]; k++) {
        size_t j = mat->col_indices[k];
        size_t l = new_row_indices[j];

        new_col_indices[l] = i;
        new_values[l] = mat->values[k];

        new_row_indices[j]++;
      }
    }

    // Shift column pointers back
    for (size_t j = mat->cols; j > 0; j--) {
      new_row_indices[j] = new_row_indices[j - 1];
    }
    new_row_indices[0] = 0;

    // Update matrix struct
    free(mat->row_indices);
    free(mat->col_indices);
    free(mat->values);
    mat->row_indices = new_row_indices;
    mat->col_indices = new_col_indices;
    mat->values = new_values;
    mat->format = CSC;
  } else {
    // Unsupported format
    fprintf(stderr, "Error: Unsupported format\n");
    exit(EXIT_FAILURE);
  }
}

// convert SparseMatrix to CSR format:
void sp_convert_to_coo(SparseMatrix *mat) {
  if (mat->format == COO) {
    // Already in COO format, nothing to do
    return;
  } else if (mat->format == CSR || mat->format == CSC) {
    // Allocate memory for new arrays
    size_t *new_row_indices = calloc(mat->nnz, sizeof(size_t));
    size_t *new_col_indices = calloc(mat->nnz, sizeof(size_t));
    double *new_values = calloc(mat->nnz, sizeof(double));

    // Copy values to new arrays
    size_t k = 0;
    for (size_t i = 0; i < mat->rows; i++) {
      for (size_t j = mat->row_indices[i]; j < mat->row_indices[i + 1]; j++) {
        new_row_indices[k] = i;
        new_col_indices[k] = mat->col_indices[j];
        new_values[k] = mat->values[j];
        k++;
      }
    }

    // Update matrix struct
    sp_destroy(mat);
    mat->row_indices = new_row_indices;
    mat->col_indices = new_col_indices;
    mat->values = new_values;
    mat->format = COO;
  } else {
    // Unsupported format
    fprintf(stderr, "Error: Unsupported format\n");
    exit(EXIT_FAILURE);
  }
}

// convert SparseMatrix to CSR format:
void sp_convert_to_csr(SparseMatrix *mat) {
  if (mat->format == CSR) {
    // Already in CSR format, nothing to do
    return;
  } else if (mat->format == COO || mat->format == CSC) {
    // Allocate memory for new arrays
    size_t *new_row_indices = calloc((mat->rows + 1), sizeof(size_t));
    size_t *new_col_indices = calloc(mat->nnz, sizeof(size_t));
    double *new_values = calloc(mat->nnz, sizeof(double));

    // Count number of non-zero elements per row
    for (size_t i = 0; i < mat->nnz; i++) {
      new_row_indices[mat->row_indices[i] + 1]++;
    }

    // Compute cumulative sum of non-zero elements per row
    for (size_t i = 1; i <= mat->rows; i++) {
      new_row_indices[i] += new_row_indices[i - 1];
    }

    // Copy values to new arrays
    for (size_t i = 0; i < mat->rows; i++) {
      for (size_t j = mat->row_indices[i]; j < mat->row_indices[i + 1]; j++) {
        size_t k = new_row_indices[i]++;
        new_col_indices[k] = mat->col_indices[j];
        new_values[k] = mat->values[j];
      }
    }

    // Shift row indices by one
    for (size_t i = mat->rows; i > 0; i--) {
      new_row_indices[i] = new_row_indices[i - 1];
    }
    new_row_indices[0] = 0;

    // Update matrix struct
    free(mat->row_indices);
    free(mat->col_indices);
    free(mat->values);
    mat->row_indices = new_row_indices;
    mat->col_indices = new_col_indices;
    mat->values = new_values;
    mat->format = CSR;
  } else {
    // Unsupported format
    fprintf(stderr, "Error: Unsupported format\n");
    exit(EXIT_FAILURE);
  }
}

// convert SparseMatrix of COO to dense format:
void sp_convert_to_dense(SparseMatrix *mat) {
  if (mat->format == DENSE) {
    // Already in dense format, nothing to do
    return;
  } else {
    // Allocate memory for dense matrix
    DoubleMatrix *dense = dm_create(mat->rows, mat->cols);

    // Copy values to dense matrix
    for (size_t i = 0; i < mat->nnz; i++) {
      size_t row = mat->row_indices[i];
      size_t col = mat->col_indices[i];
      double val = mat->values[i];
      dense->values[row * mat->cols + col] = val;
    }

    // Free memory of sparse matrix
    sp_destroy(mat);

    // Update matrix struct
    mat->format = DENSE;
    mat->row_indices = NULL;
    mat->col_indices = NULL;
    mat->values = dense->values;
  }
}

// convert DoubleMatrix to sparse(COO-)format:
void sp_convert_to_sparse(DoubleMatrix *mat) {
  if (mat->format != DENSE) {
    // Already in sparse format, nothing to do
    return;
  } else {
    // Allocate memory for sparse matrix
    SparseMatrix *sparse = sp_create(mat->rows, mat->cols);

    // Loop through dense matrix to find non-zero elements
    for (size_t row = 0; row < mat->rows; row++) {
      for (size_t col = 0; col < mat->cols; col++) {
        double val = mat->values[row * mat->cols + col];
        if (val != 0) {
          sparse->row_indices[sparse->nnz] = row;
          sparse->col_indices[sparse->nnz] = col;
          sparse->values[sparse->nnz] = val;
          sparse->nnz++;
        }
      }
    }

    // Reallocate memory for row indices, column indices, and values arrays
    sparse->row_indices =
        (size_t *)realloc(sparse->row_indices, sparse->nnz * sizeof(size_t));
    sparse->col_indices =
        (size_t *)realloc(sparse->col_indices, sparse->nnz * sizeof(size_t));
    sparse->values =
        (double *)realloc(sparse->values, sparse->nnz * sizeof(double));

    // Free memory of dense matrix
    dm_destroy(mat);

    // Update matrix struct
    mat->format = COO;
    mat->row_indices = sparse->row_indices;
    mat->col_indices = sparse->col_indices;
    mat->values = sparse->values;
    mat->nnz = sparse->nnz;

    // Free memory of sparse matrix
    sp_destroy(sparse);
  }
}
