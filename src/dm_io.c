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
#include <assert.h>
#include <ncurses.h>

// #define NDEBUG
#define BRAILLE_SIZE 10

/*******************************/
/*         I/O Functions       */
/*******************************/
#define MAX_ROW 20
#define MAX_ROW_PRINT 5
#define MAX_COLUMN 10
#define MAX_COLUMN_PRINT 4

/*******************************/
/*            Vector           */
/*******************************/

/* read in DoubleVector data from file */
int dv_read_from_file(DoubleVector *vec, const char *filepath) {
  FILE *file = fopen(filepath, "r");
  if (file == NULL) {
    return 1;
  }
  int succ_read = 1;
  for (size_t i = 0; i < vec->rows; i++) {
    // FIXME: insecure use of fscanf:
    succ_read = fscanf(file, "%15lf", &vec->values[i]);
  }
  fclose(file);

  return succ_read;
}

/* write data from DoubleVector to file */
int dv_write_to_file(DoubleVector *vec, const char *filepath) {
  FILE *file = fopen(filepath, "w");
  if (file == NULL) {
    return 1;
  }

  for (size_t i = 0; i < vec->rows; i++) {
    if (i < vec->rows - 1) {
      fprintf(file, "%lf\n", vec->values[i]);
    } else {
      fprintf(file, "%lf", vec->values[i]);
    }
  }
  fclose(file);

  return 0;
}

/**
 * @brief printf DoubleArray pretty
 *
 * @param DoubleVector* vec
 */
void dv_print(const DoubleVector *vec) {
  if (dv_is_row_vector(vec)) {
    dv_print_row(vec);
  } else if (dv_is_row_vector(vec) == false) {
    dv_print_col(vec);
  } else {
    printf("not a vector!\n");
  }
}

// function to print DOubleVector as row vector
static void dv_print_row(const DoubleVector *vec) {
  if (dv_is_row_vector(vec)) {
    printf("[ ");
    for (size_t i = 0; i < vec->rows; i++) {
      if (i > 0) {
        printf(", ");
      }
      printf("%.2lf", vec->values[i]);
    }
    printf(" ]\n");
  }
}

// function to print DOubleVector as column vector  (transposed)
static void dv_print_col(const DoubleVector *vec) {
  if (dv_is_row_vector(vec) == false) {
    for (size_t i = 0; i < vec->cols; i++) {
      printf("[ %.2lf ]\n", vec->values[i]);
    }
    printf("\n");
  }
}

/*******************************/
/*            Matrix           */
/*******************************/

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
  printf("Matrix %zix%zi\n", matrix->rows, matrix->cols);
}

/**
 * @brief printf SparseMatrix pretty in braille form
 *
 * @param matrix
 */
void sp_print_braille(const SparseMatrix *mat) {
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
  printf("SparseMatrix (%zu x %zu)\n", mat->rows, mat->cols);
}

// print all fields of a SparseMatrix *mat
void sp_print(const SparseMatrix *mat) {
  printf("SparseMatrix (%zu x %zu)\n", mat->rows, mat->cols);
  printf("values: ");
  for (size_t i = 0; i < mat->nnz; i++) {
    printf("%lf ", mat->values[i]);
  }
  printf("\n");
  printf("row_indices: ");
  for (size_t i = 0; i < mat->rows + 1; i++) {
    printf("%zu ", mat->row_indices[i]);
  }
  printf("\n");
  printf("col_indices: ");
  for (size_t i = 0; i < mat->nnz; i++) {
    printf("%zu ", mat->col_indices[i]);
  }
  printf("\n");
}

void sp_print_condensed(SparseMatrix *mat) {
  printf("SparseMatrix (%zu x %zu)\n", mat->rows, mat->cols);
  for (size_t i = 0; i < mat->rows; i++) {
    size_t start = mat->row_indices[i];
    size_t end = mat->row_indices[i + 1];

    for (size_t j = start; j < end; j++) {
      printf("(%zu,%zu): %f ", i, mat->col_indices[j], mat->values[j]);
    }

    printf("\n");
  }
}

void sp_create_scatterplot(const SparseMatrix *mat, const char *filename) {

  StringReference *errorMessage;
  RGBABitmapImageReference *imageReference = CreateRGBABitmapImageReference();

  int *xs = (int *)mat->row_indices;
  int *ys = (int *)mat->col_indices;

  // Create the plot
  ScatterPlotSeries *series = GetDefaultScatterPlotSeriesSettings();
  series->xs = (double *)xs;
  series->xsLength = mat->nnz / sizeof(double);
  series->ys = (double *)ys;
  series->ysLength = mat->rows / sizeof(double);
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
    size_t length;
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
