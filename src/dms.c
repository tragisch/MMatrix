#include "dms.h"

DoubleSparseMatrix *dms_create_test_matrix(size_t rows, size_t cols, size_t nnz,
                                           size_t *row_indices,
                                           size_t *col_indices,
                                           double *values) {
  DoubleSparseMatrix *mat = dms_create(rows, cols, nnz);
  for (size_t i = 0; i < nnz; i++) {
    mat->row_indices[i] = row_indices[i];
    mat->col_indices[i] = col_indices[i];
    mat->values[i] = values[i];
  }
  mat->nnz = nnz;
  return mat;
}

DoubleSparseMatrix *dms_create(size_t rows, size_t cols, size_t nnz) {
  if (rows < 1 || cols < 1) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }

  DoubleSparseMatrix *mat = malloc(sizeof(DoubleSparseMatrix));
  if (!mat) {
    perror("Error allocating memory for matrix struct");
    return NULL;
  }

  mat->rows = rows;
  mat->cols = cols;
  mat->nnz = nnz;
  mat->capacity = INIT_CAPACITY;

  mat->row_indices = calloc(dms_max_int(nnz, mat->capacity), sizeof(size_t));
  if (!mat->row_indices) {
    perror("Error allocating memory for row indices");
    free(mat);
    return NULL;
  }

  mat->col_indices = calloc(dms_max_int(nnz, mat->capacity), sizeof(size_t));
  if (!mat->col_indices) {
    perror("Error allocating memory for column indices");
    free(mat->row_indices);
    free(mat);
    return NULL;
  }

  mat->values = calloc(dms_max_int(nnz, mat->capacity), sizeof(double));
  if (!mat->values) {
    perror("Error allocating memory for values");
    free(mat->col_indices);
    free(mat->row_indices);
    free(mat);
    return NULL;
  }

  return mat;
}

DoubleSparseMatrix *dms_clone(const DoubleSparseMatrix *m) {
  DoubleSparseMatrix *copy = dms_create(m->rows, m->cols, m->nnz);
  for (size_t i = 0; i < m->nnz; i++) {
    copy->row_indices[i] = m->row_indices[i];
    copy->col_indices[i] = m->col_indices[i];
    copy->values[i] = m->values[i];
  }
  return copy;
}

DoubleSparseMatrix *dms_identity(size_t n) {
  if (n < 1) {
    perror("Error: invalid identity dimensions.\n");
    return NULL;
  }
  DoubleSparseMatrix *mat = dms_create(n, n, n);
  for (size_t i = 0; i < n; i++) {
    mat->row_indices[i] = i;
    mat->col_indices[i] = i;
    mat->values[i] = 1.0;
  }
  return mat;
}

cs *dms_to_cs(const DoubleSparseMatrix *coo) {
  int m = coo->rows;
  int n = coo->cols;
  int nz = coo->nnz;

  // Allocate a CSparse matrix in COO format
  cs *T = cs_spalloc(m, n, nz, 1, 1);
  if (!T)
    return NULL;

  // Fill the CSparse matrix with the data from the DoubleSparseMatrix
  for (size_t k = 0; k < nz; k++) {
    cs_entry(T, coo->row_indices[k], coo->col_indices[k], coo->values[k]);
  }

  // Convert the COO matrix to CSC format
  cs *A = cs_compress(T);
  cs_spfree(T); // Free the temporary COO matrix

  return A;
}

DoubleSparseMatrix *cs_to_dms(const cs *A) {
  // Allocate memory for the DoubleSparseMatrix structure
  DoubleSparseMatrix *coo =
      (DoubleSparseMatrix *)malloc(sizeof(DoubleSparseMatrix));
  if (!coo)
    return NULL;

  coo->rows = A->m;
  coo->cols = A->n;
  coo->nnz = A->nzmax;
  coo->capacity = A->nzmax + INIT_CAPACITY;

  // Allocate memory for the COO arrays
  coo->row_indices =
      (size_t *)malloc(dms_max_int(coo->nnz, coo->capacity) * sizeof(size_t));
  coo->col_indices =
      (size_t *)malloc(dms_max_int(coo->nnz, coo->capacity) * sizeof(size_t));
  coo->values =
      (double *)malloc(dms_max_int(coo->nnz, coo->capacity) * sizeof(double));

  if (!coo->row_indices || !coo->col_indices || !coo->values) {
    free(coo->row_indices);
    free(coo->col_indices);
    free(coo->values);
    free(coo);
    return NULL;
  }

  // Fill the COO arrays with the data from the CSC matrix
  size_t nnz_index = 0;
  for (size_t col = 0; col < A->n; col++) {
    for (size_t p = A->p[col]; p < A->p[col + 1]; p++) {
      coo->row_indices[nnz_index] = A->i[p];
      coo->col_indices[nnz_index] = col;
      coo->values[nnz_index] = A->x[p];
      nnz_index++;
    }
  }

  return coo;
}

DoubleSparseMatrix *dms_rand(size_t rows, size_t cols, double density) {
  if (density < 0.0 || density > 1.0) {
    perror("Error: invalid density value.\n");
    return NULL;
  }

  size_t nnz = (size_t)(rows * cols * density);
  if (nnz < EPSILON) {
    DoubleSparseMatrix *mat = dms_create(rows, cols, 0.0);
    return mat;
  }

  DoubleSparseMatrix *mat = dms_create(rows, cols, nnz);
  for (size_t i = 0; i < nnz; i++) {
#ifdef __APPLE__
    mat->row_indices[i] = arc4random_uniform((uint32_t)rows);
    mat->col_indices[i] = arc4random_uniform((uint32_t)cols);
    mat->values[i] = (double)arc4random() / UINT32_MAX;
#else
    mat->row_indices[i] = rand() % rows;
    mat->col_indices[i] = rand() % cols;
    mat->values[i] = (double)rand() / RAND_MAX;
#endif
  }
  return mat;
}

DoubleSparseMatrix *dms_convert_array(size_t rows, size_t cols,
                                      double array[rows][cols]) {
  size_t nnz = 0;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (array[i][j] != 0) {
        nnz++;
      }
    }
  }
  DoubleSparseMatrix *mat = dms_create(rows, cols, nnz);
  size_t k = 0;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (array[i][j] != 0) {
        mat->row_indices[k] = i;
        mat->col_indices[k] = j;
        mat->values[k] = array[i][j];
        k++;
      }
    }
  }
  return mat;
}

DoubleSparseMatrix *dms_get_row(const DoubleSparseMatrix *mat, size_t i) {
  if (i >= mat->rows) {
    perror("Error: invalid row index.\n");
    return NULL;
  }

  DoubleSparseMatrix *row = dms_create(1, mat->cols, mat->cols);
  size_t k = 0;
  for (size_t j = 0; j < mat->nnz; j++) {
    if (mat->row_indices[j] == i) {
      row->row_indices[k] = 0;
      row->col_indices[k] = mat->col_indices[j];
      row->values[k] = mat->values[j];
      k++;
    }
  }
  row->nnz = k;
  return row;
}

DoubleSparseMatrix *dms_get_last_row(const DoubleSparseMatrix *mat) {
  return dms_get_row(mat, mat->rows - 1);
}

DoubleSparseMatrix *dms_get_col(const DoubleSparseMatrix *mat, size_t j) {
  DoubleSparseMatrix *col = dms_create(mat->rows, 1, mat->rows);
  size_t k = 0;
  for (size_t i = 0; i < mat->nnz; i++) {
    if (mat->col_indices[i] == j) {
      col->row_indices[k] = mat->row_indices[i];
      col->col_indices[k] = 0;
      col->values[k] = mat->values[i];
      k++;
    }
  }
  col->nnz = k;
  return col;
}

DoubleSparseMatrix *dms_get_last_col(const DoubleSparseMatrix *mat) {
  return dms_get_col(mat, mat->cols - 1);
}

DoubleSparseMatrix *dms_multiply(const DoubleSparseMatrix *mat1,
                                 const DoubleSparseMatrix *mat2) {
  if (mat1->cols != mat2->rows) {
    perror("Error: invalid matrix dimensions.\n");
    return NULL;
  }
  // use cs_multiply from csparse
  cs *A = dms_to_cs(mat1);
  cs *B = dms_to_cs(mat2);
  cs *C = cs_multiply(A, B);
  DoubleSparseMatrix *result = cs_to_dms(C);

  cs_spfree(A);
  cs_spfree(B);
  cs_spfree(C);

  return result;
}

DoubleSparseMatrix *dms_multiply_by_number(const DoubleSparseMatrix *mat,
                                           const double number) {
  DoubleSparseMatrix *result = dms_clone(mat);
  for (size_t i = 0; i < mat->nnz; i++) {
    result->values[i] *= number;
  }
  return result;
}

DoubleSparseMatrix *dms_transpose(const DoubleSparseMatrix *mat) {
  if (mat->nnz == 0) {
    return dms_create(mat->cols, mat->rows, 0);
  }
  // use cs_transpose from csparse
  cs *A = dms_to_cs(mat);
  cs *AT = cs_transpose(A, 1);
  DoubleSparseMatrix *result = cs_to_dms(AT);
  return result;
}

double dms_get(const DoubleSparseMatrix *mat, size_t i, size_t j) {
  if (i >= mat->rows || j >= mat->cols) {
    perror("Error: matrix index out of bounds.\n");
    return 0.0;
  }
  for (size_t k = 0; k < mat->nnz; k++) {
    if (mat->row_indices[k] == i && mat->col_indices[k] == j) {
      return mat->values[k];
    }
  }
  return 0.0;
}

void dms_set(DoubleSparseMatrix *matrix, size_t i, size_t j, double value) {
  // Find the position of the element (i, j) in the matrix
  size_t position = _dms_binary_search(matrix, i, j);

  if (position < matrix->nnz && matrix->row_indices[position] == i &&
      matrix->col_indices[position] == j) {
    // Element already exists at position (i, j), update the value
    matrix->values[position] = value;
  } else {
    _dms_insert_element(matrix, i, j, value, position);
  }
}

static size_t _dms_binary_search(const DoubleSparseMatrix *matrix, size_t i,
                                 size_t j) {
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

static void _dms_insert_element(DoubleSparseMatrix *matrix, size_t i, size_t j,
                                double value, size_t position) {
  // Increase the capacity if needed
  if (matrix->nnz == matrix->capacity) {
    dms_realloc(matrix, matrix->capacity * 2);
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

void dms_realloc(DoubleSparseMatrix *mat, size_t new_capacity) {

  if (new_capacity <= mat->capacity) {
    printf("Can not resize matrix to smaller capacity!\n");
    exit(EXIT_FAILURE);
  }

  if (new_capacity == 0) {
    new_capacity = mat->capacity * 2;
  }

  // resize matrix:
  size_t *row_indices =
      (size_t *)realloc(mat->row_indices, (new_capacity) * sizeof(size_t));
  size_t *col_indices =
      (size_t *)realloc(mat->col_indices, (new_capacity) * sizeof(size_t));
  double *values =
      (double *)realloc(mat->values, (new_capacity) * sizeof(double));
  if (row_indices == NULL || col_indices == NULL || values == NULL) {
    printf("Error allocating memory!\n");
    exit(EXIT_FAILURE);
  }

  mat->capacity = new_capacity;
  mat->row_indices = row_indices;
  mat->col_indices = col_indices;
  mat->values = values;
}

void dms_print(const DoubleSparseMatrix *mat) {
  if (mat->nnz == 0) {
    printf("Empty matrix\n");
    return;
  }
  printf("values: [");
  for (size_t i = 0; i < mat->nnz; i++) {
    printf("%.2lf, ", mat->values[i]);
  }
  printf("]\n");

  printf("row_indices: [");
  if (mat->row_indices != NULL) {
    for (size_t i = 0; i < mat->nnz; i++) {
      printf("%zu, ", mat->row_indices[i]);
    }
  }
  printf("]\n");

  printf("col_indices: [");
  if (mat->col_indices != NULL) {
    for (size_t i = 0; i < mat->nnz; i++) {
      printf("%zu, ", mat->col_indices[i]);
    }
  }

  printf("]\n");
}

void dms_destroy(DoubleSparseMatrix *mat) {
  free(mat->row_indices);
  free(mat->col_indices);
  free(mat->values);
  free(mat);
}

double dms_max_double(double a, double b) { return a > b ? a : b; }
double dms_min_double(double a, double b) { return a < b ? a : b; }
int dms_max_int(int a, int b) { return a > b ? a : b; }