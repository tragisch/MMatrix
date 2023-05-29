/**
 * @file dm_matrix_market.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm.h"
#include "dm_io.h"
#include "dm_modify.h"
#include "dm_utils.h"

/*******************************/
/*     Matrix Market Format    */
/*******************************/

/**
 * @brief Read a Matrix Market file and return a DoubleMatrix in Sparse format
 * format
 *
 * @param filename
 * @return DoubleMatrix*
 *
 * */
DoubleMatrix *dm_read_matrix_market(const char *filename) {
  FILE *fp = NULL;
  size_t nrows = 0;
  size_t ncols = 0;
  size_t nnz = 0;

  // Open the Matrix Market file for reading
  fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Error: Unable to open file.\n");
    exit(1);
  }

  char line[1024];
  fgets(line, sizeof(line), fp);
  if (strstr(line, "%%MatrixMarket matrix coordinate") == NULL) {
    perror("Error: invalid header. No MatrixMarket matrix coordinate.\n");
  }

  // Skip all comment lines in the file
  while (fgets(line, 1024, fp) != NULL) {
    if (line[0] != '%') {
      break;
    }
  }

  // Read dimensions and number of non-zero values
  sscanf(line, "%zu %zu %zu", &nrows, &ncols, &nnz);

  // Create DoubleMatrix
  DoubleMatrix *mat = dm_create_nnz(nrows, ncols, nnz);

  if (nnz > 500) {
    printf("Reading Matrix Market file: %s\n", filename);
  }

  // Read non-zero values
  for (size_t i = 0; i < nnz; i++) {
    if (nnz > 500) {
      print_progress_bar(i, nnz, 50);
    }
    size_t row_idx = 0;
    size_t col_idx = 0;
    double val = 0.0;

    fscanf(fp, "%zu %zu %lf", &row_idx, &col_idx, &val);

    if (val != 0.0) {
      mat->row_indices[i] = (size_t)(row_idx - 1);
      mat->col_indices[i] = (size_t)(col_idx - 1);
      mat->values[i] = (double)val;
      mat->nnz++;
    }
  }

  if (nnz > 500) {
    printf("\n");
  }

  // Close the file
  fclose(fp);

  dm_cleanup(mat);

  return mat;
}

/**
 * @brief Write a DoubleMatrix to a Matrix Market file
 *
 * @param mat
 * @param filename
 */
void dm_write_matrix_market(const DoubleMatrix *mat, const char *filename) {
  FILE *fp = NULL;
  fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("Error: Unable to open file.\n");
    exit(1);
  }

  fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(fp, "%zu %zu %zu\n", mat->rows, mat->cols, mat->rows * mat->cols);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      fprintf(fp, "%zu %zu %lf\n", i + 1, j + 1, dm_get(mat, i, j));
    }
  }

  fclose(fp);
}
