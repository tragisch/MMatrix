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
SparseMatrix *sp_laplace(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = 2 * n - 1;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    mat->values[i] = 2;
    mat->row_indices[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < n - 1; i++) {
    mat->values[i + n] = -1;
    mat->row_indices[i + n] = i;
    mat->col_indices[i + n] = i + 1;
    mat->values[i + n + 1] = -1;
    mat->row_indices[i + n + 1] = i + 1;
    mat->col_indices[i + n + 1] = i;
  }
  return mat;
}

// Hilbert Matrix
SparseMatrix *sp_hilbert(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = 1.0 / (i + j + 1);
      mat->row_indices[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// Vandermonde Matrix
SparseMatrix *sp_vandermonde(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = pow(i + 1, j);
      mat->row_indices[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// Toeplitz Matrix
SparseMatrix *sp_toeplitz(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = 2 * n - 1;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    mat->values[i] = 2;
    mat->row_indices[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < n - 1; i++) {
    mat->values[i + n] = -1;
    mat->row_indices[i + n] = i;
    mat->col_indices[i + n] = i + 1;
    mat->values[i + n + 1] = -1;
    mat->row_indices[i + n + 1] = i + 1;
    mat->col_indices[i + n + 1] = i;
  }
  return mat;
}

// Circulant Matrix
SparseMatrix *sp_circulant(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = 2 * n - 1;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    mat->values[i] = 2;
    mat->row_indices[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < n - 1; i++) {
    mat->values[i + n] = -1;
    mat->row_indices[i + n] = i;
    mat->col_indices[i + n] = i + 1;
    mat->values[i + n + 1] = -1;
    mat->row_indices[i + n + 1] = i + 1;
    mat->col_indices[i + n + 1] = i;
  }
  return mat;
}

// Hankel Matrix
SparseMatrix *sp_hankel(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = 2 * n - 1;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    mat->values[i] = 2;
    mat->row_indices[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < n - 1; i++) {
    mat->values[i + n] = -1;
    mat->row_indices[i + n] = i;
    mat->col_indices[i + n] = i + 1;
    mat->values[i + n + 1] = -1;
    mat->row_indices[i + n + 1] = i + 1;
    mat->col_indices[i + n + 1] = i;
  }
  return mat;
}

// DFT Matrix
SparseMatrix *sp_dft(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = cos(2 * M_PI * i * j / n);
      mat->row_indices[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// DCT Matrix
SparseMatrix *sp_dct(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = cos(M_PI * (i + 0.5) * j / n);
      mat->row_indices[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// DST Matrix
SparseMatrix *sp_dst(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = sin(M_PI * (i + 0.5) * j / n);
      mat->row_indices[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

// adajacent matrix
SparseMatrix *sp_adjacent(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = 2 * n - 2;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n - 1; i++) {
    mat->values[i] = 1;
    mat->row_indices[i] = i;
    mat->col_indices[i] = i + 1;
    mat->values[i + n - 1] = 1;
    mat->row_indices[i + n - 1] = i + 1;
    mat->col_indices[i + n - 1] = i;
  }
  return mat;
}

SparseMatrix *sp_hadamard(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = 1;
      mat->row_indices[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

SparseMatrix *sp_kronecker(size_t n) {
  SparseMatrix *mat = sp_create(n, n);
  mat->nnz = n * n;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat->values[i * n + j] = 1;
      mat->row_indices[i * n + j] = i;
      mat->col_indices[i * n + j] = j;
    }
  }
  return mat;
}

SparseMatrix *sp_diagonal_matrix(SparseMatrix *adj) {
  SparseMatrix *mat = sp_create(adj->rows, adj->cols);
  mat->nnz = adj->rows;
  mat->values = malloc(mat->nnz * sizeof(double));
  mat->row_indices = malloc(mat->nnz * sizeof(size_t));
  mat->col_indices = malloc(mat->nnz * sizeof(size_t));
  for (int i = 0; i < adj->rows; i++) {
    mat->values[i] = 0;
    mat->row_indices[i] = i;
    mat->col_indices[i] = i;
  }
  for (int i = 0; i < adj->nnz; i++) {
    mat->values[adj->row_indices[i]] += 1;
  }
  return mat;
}

void sp_to_laplace(SparseMatrix *A) {
  for (int i = 0; i < A->rows; i++) {
    double sum = 0;
    for (int j = 0; j < A->nnz; j++) {
      if (A->row_indices[j] == i) {
        sum += A->values[j];
      }
    }
    for (int j = 0; j < A->nnz; j++) {
      if (A->row_indices[j] == i) {
        A->values[j] = -A->values[j] / sum;
      }
    }
    for (int j = 0; j < A->nnz; j++) {
      if (A->row_indices[j] == i && A->col_indices[j] == i) {
        A->values[j] = 1;
      }
    }
  }
}
