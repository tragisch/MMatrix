/**
 * @file dm_io.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm_io.h"
#include "dbg.h"
#include "dm_math.h"
#include "dm_matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// #define NDEBUG
#define BRAILLE_SIZE 10
enum { INIT_CAPACITY = 2U };

/*******************************/
/*         I/O Functions       */
/*******************************/
#define MAX_ROW 20
#define MAX_ROW_PRINT 5
#define MAX_COLUMN 10
#define MAX_COLUMN_PRINT 4

/*******************************/
/*     Matrix Market Format    */
/*******************************/

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
  DoubleMatrix *mat = dm_create(nrows, ncols);

  // Read non-zero values
  for (size_t i = 0; i < (nnz); i++) {
    size_t row_idx = 0;
    size_t col_idx = 0;
    double val = NAN;

    fscanf(fp, "%zu %zu %lf", &row_idx, &col_idx, &val);

    if (val != 0.0) {
      dm_set(mat, row_idx - 1, col_idx - 1, val);
    }
  }

  // Close the file
  fclose(fp);
  return mat;
}

/*******************************/
/*   Pretty print  Vector      */
/*******************************/

/**
 * @brief printf DoubleArray pretty
 *
 * @param DoubleVector* vec
 */
void dv_print(const DoubleVector *vec) {
  if (vec->rows >= vec->cols) {
    dv_print_row(vec);
  } else {
    dv_print_col(vec);
  }
}

// function to print DOubleVector as row vector
static void dv_print_row(const DoubleVector *vec) {

  if (vec->rows >= vec->cols) {
    printf("[ ");
    for (size_t i = 0; i < vec->rows; i++) {
      if (i > 0) {
        printf(", ");
      }
      printf("%.2lf", vec->values[i]);
    }
    printf(" ]\n");
  }
  printf("Vector: %zu x %zu\n", vec->rows, vec->cols);
}

// function to print DOubleVector as column vector  (transposed)
static void dv_print_col(const DoubleVector *vec) {

  if (vec->cols > vec->rows) {
    for (size_t i = 0; i < vec->cols; i++) {
      printf("[ %.2lf ]\n", vec->values[i]);
    }
    printf("\n");
  }
  printf("Vector: %zu x %zu\n", vec->rows, vec->cols);
}

/*******************************/
/*   Pretty print  Matrix      */
/*******************************/

void dm_brief(const DoubleMatrix *mat) {
  printf("Matrix: %zu x %zu\n", mat->rows, mat->cols);
  printf("Non-zero elements: %zu\n", mat->nnz);
  printf("Density: %lf\n", dm_density(mat));
}

/**
 * @brief printf DoubleMatrix pretty
 *
 * @param matrix
 */
void dm_print(const DoubleMatrix *matrix) {
  // print to console a DoubleMatrix matrix row by row with 2 digits precision
  for (size_t i = 0; i < matrix->rows; i++) {
    printf("[ ");
    for (size_t j = 0; j < matrix->cols; j++) {
      printf("%.2lf ", dm_get(matrix, i, j));
    }
    printf("]\n");
  }
  print_matrix_dimension(matrix);
}

/**
 * @brief printf SparseMatrix pretty in braille form
 *
 * @param matrix
 */
void sp_print_braille(const DoubleMatrix *mat) {
  printf("--Braille-Form: \n");
  // Define Braille characters for matrix elements
  const char *braille[] = {
      "\u2800", "\u2801", "\u2803", "\u2809", "\u2819", "\u2811", "\u281b",
      "\u2813", "\u280a", "\u281a", "\u2812", "\u281e", "\u2816", "\u2826",
      "\u2822", "\u282e", "\u281c", "\u282c", "\u2824", "\u283a", "\u2832",
      "\u2836", "\u2834", "\u283e", "\u2818", "\u2828", "\u2820", "\u2838",
      "\u2830", "\u283c", "\u283a", "\u283e", "\u2807", "\u280f", "\u2817",
      "\u281f", "\u280e", "\u281e", "\u2827", "\u2837", "\u282f", "\u280b",
      "\u281b", "\u2823", "\u283b", "\u2833", "\u2837", "\u283f", "\u281d",
      "\u282d", "\u2825", "\u283d", "\u2835", "\u283f", "\u283b", "\u283f",
      "\u2806", "\u280c", "\u2814", "\u281a", "\u282c", "\u2832", "\u283a",
      "\u283e"};

  // Print the matrix
  size_t idx = 0;
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      if (j == mat->col_indices[idx]) {
        int val = (int)round(mat->values[idx] * 4.0);
        printf("%s", braille[val]);
        idx++;
      } else {
        printf("%s", " ");
      }
    }

    printf("\n");
  }
  print_matrix_dimension(mat);
}

// print all fields of a SparseMatrix *mat
void sp_print(const DoubleMatrix *mat) {
  print_matrix_dimension(mat);
  printf("values: ");
  for (size_t i = 0; i < mat->nnz; i++) {
    printf("%.2lf ", mat->values[i]);
  }

  printf("\n");
  printf("row_indices: ");
  if (mat->row_indices != NULL) {
    for (size_t i = 0; i < mat->nnz; i++) {
      printf("%zu ", mat->row_indices[i]);
    }
  }

  printf("\n");
  printf("col_indices: ");
  if (mat->col_indices != NULL) {
    for (size_t i = 0; i < mat->nnz; i++) {
      printf("%zu ", mat->col_indices[i]);
    }
  }
  printf("\n");
}

void sp_print_condensed(DoubleMatrix *mat) {
  print_matrix_dimension(mat);
  size_t start = mat->row_indices[0];
  for (size_t i = 0; i < mat->nnz; i++) {
    if (start != mat->row_indices[i]) {
      printf("\n");
      start = mat->row_indices[i];
    }
    printf("(%zu,%zu): %.2lf, ", mat->row_indices[i], mat->col_indices[i],
           mat->values[i]);
  }
  printf("\n");
}

void sp_create_scatterplot(const DoubleMatrix *mat, const char *filename) {

  StringReference *errorMessage = NULL;
  RGBABitmapImageReference *imageReference = CreateRGBABitmapImageReference();

  double *xs = malloc(sizeof(double) * mat->nnz);
  double *ys = malloc(sizeof(double) * mat->nnz);

  // convert mat->row_pointers to double array

  for (size_t i = 0; i < mat->nnz; i++) {
    xs[i] = (double)mat->row_indices[i];
    ys[i] = (double)mat->col_indices[i];
  }

  dbga(xs, mat->nnz);
  dbga(ys, mat->nnz);

  // Create the plot
  ScatterPlotSeries *series = GetDefaultScatterPlotSeriesSettings();
  series->xs = xs;
  series->xsLength = mat->nnz;
  series->ys = ys;
  series->ysLength = mat->rows;
  series->linearInterpolation = false;
  series->pointType = L"dots";
  series->pointTypeLength = wcslen(series->pointType);
  series->color = CreateRGBColor(1, 0, 0);

  ScatterPlotSettings *settings = GetDefaultScatterPlotSettings();
  settings->width = 600;
  settings->height = 400;
  settings->autoBoundaries = true;
  settings->autoPadding = true;
  settings->title = L"";
  settings->titleLength = wcslen(settings->title);
  settings->xLabel = L"";
  settings->xLabelLength = wcslen(settings->xLabel);
  settings->yLabel = L"";
  settings->yLabelLength = wcslen(settings->yLabel);

  ScatterPlotSeries *s[] = {series};
  settings->scatterPlotSeries = s;
  settings->scatterPlotSeriesLength = 1;

  errorMessage = (StringReference *)malloc(sizeof(StringReference));
  bool success =
      DrawScatterPlotFromSettings(imageReference, settings, errorMessage);

  if (success) {
    size_t length = 0;
    double *pngdata = ConvertToPNG(&length, imageReference->image);
    WriteToFile(pngdata, length, "matrix_scatterplot.png");
    DeleteImage(imageReference->image);
  } else {
    fprintf(stderr, "Error: ");
    for (int i = 0; i < errorMessage->stringLength; i++) {
      fprintf(stderr, "%c", errorMessage->string[i]);
    }
    fprintf(stderr, "\n");
  }
}

static void print_matrix_dimension(const DoubleMatrix *mat) {
  switch (mat->format) {
  case SPARSE:
    printf("SparseMatrix (%zu x %zu)\n", mat->rows, mat->cols);
    break;
  case DENSE:
    printf("DenseMatrix (%zu x %zu)\n", mat->rows, mat->cols);
    break;
  case VECTOR:
    printf("Vector (%zu x %zu)\n", mat->rows, mat->cols);
    break;
  }
}
