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

#include "dbg.h"
#include "dm_math.h"
#include "dm_matrix.h"

/*******************************/
/*        Special Matrix        */
/*******************************/

// Laplace Matrix
DoubleMatrix *sp_laplace(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = 2 * n - 1;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    mat->values[i] = 2;
    mat->row_pointers[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < n - 1; i++) {
    mat->values[i + n] = -1;
    mat->row_pointers[i + n] = i;
    mat->col_indices[i + n] = i + 1;
    mat->values[i + n + 1] = -1;
    mat->row_pointers[i + n + 1] = i + 1;
    mat->col_indices[i + n + 1] = i;
  }
  return mat;
}

// Hilbert Matrix
DoubleMatrix *sp_hilbert(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = 1.0 / (i + j + 1);
      mat->row_pointers[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// Vandermonde Matrix
DoubleMatrix *sp_vandermonde(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = pow(i + 1, j);
      mat->row_pointers[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// Toeplitz Matrix
DoubleMatrix *sp_toeplitz(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = 2 * n - 1;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    mat->values[i] = 2;
    mat->row_pointers[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < n - 1; i++) {
    mat->values[i + n] = -1;
    mat->row_pointers[i + n] = i;
    mat->col_indices[i + n] = i + 1;
    mat->values[i + n + 1] = -1;
    mat->row_pointers[i + n + 1] = i + 1;
    mat->col_indices[i + n + 1] = i;
  }
  return mat;
}

// Circulant Matrix
DoubleMatrix *sp_circulant(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = 2 * n - 1;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    mat->values[i] = 2;
    mat->row_pointers[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < n - 1; i++) {
    mat->values[i + n] = -1;
    mat->row_pointers[i + n] = i;
    mat->col_indices[i + n] = i + 1;
    mat->values[i + n + 1] = -1;
    mat->row_pointers[i + n + 1] = i + 1;
    mat->col_indices[i + n + 1] = i;
  }
  return mat;
}

// Hankel Matrix
DoubleMatrix *sp_hankel(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = 2 * n - 1;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    mat->values[i] = 2;
    mat->row_pointers[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < n - 1; i++) {
    mat->values[i + n] = -1;
    mat->row_pointers[i + n] = i;
    mat->col_indices[i + n] = i + 1;
    mat->values[i + n + 1] = -1;
    mat->row_pointers[i + n + 1] = i + 1;
    mat->col_indices[i + n + 1] = i;
  }
  return mat;
}

// DFT Matrix
DoubleMatrix *sp_dft(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = cos(2 * M_PI * i * j / n);
      mat->row_pointers[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// DCT Matrix
DoubleMatrix *sp_dct(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = cos(M_PI * (i + 0.5) * j / n);
      mat->row_pointers[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// DST Matrix
DoubleMatrix *sp_dst(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = sin(M_PI * (i + 0.5) * j / n);
      mat->row_pointers[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// adajacent matrix
DoubleMatrix *sp_adjacent(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = 2 * n - 2;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n - 1; i++) {
    mat->values[i] = 1;
    mat->row_pointers[i] = i;
    mat->col_indices[i] = i + 1;
    mat->values[i + n - 1] = 1;
    mat->row_pointers[i + n - 1] = i + 1;
    mat->col_indices[i + n - 1] = i;
  }
  return mat;
}

DoubleMatrix *sp_hadamard(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = 1;
      mat->row_pointers[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

DoubleMatrix *sp_kronecker(size_t n) {
  DoubleMatrix *mat = dm_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = 1;
      mat->row_pointers[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

DoubleMatrix *sp_diagonal_matrix(DoubleMatrix *adj) {
  DoubleMatrix *mat = dm_create(adj->rows, adj->cols);
  mat->nnz = adj->rows;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_pointers = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < adj->rows; i++) {
    mat->values[i] = 0;
    mat->row_pointers[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < adj->nnz; i++) {
    mat->values[adj->row_pointers[i]] += 1;
  }
  return mat;
}

void sp_to_laplace(DoubleMatrix *A) {
  for (int i = 0; i < A->rows; i++) {
    double sum = 0;
    for (int j = 0; j < A->nnz; j++) {
      if (A->row_pointers[j] == i) {
        sum += A->values[j];
      }
    }
    for (int j = 0; j < A->nnz; j++) {
      if (A->row_pointers[j] == i) {
        A->values[j] = -A->values[j] / sum;
      }
    }
    for (int j = 0; j < A->nnz; j++) {
      if (A->row_pointers[j] == i && A->col_indices[j] == i) {
        A->values[j] = 1;
      }
    }
  }
}
