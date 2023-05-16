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
#include "dm.h"
#include "dm_internals.h"
#include "dm_math.h"

#define BRAILLE_SIZE 10
enum { INIT_CAPACITY = 2U };

/*******************************/
/*         I/O Functions       */
/*******************************/
#define MAX_ROW 20
#define MAX_ROW_PRINT 5
#define MAX_COLUMN 10
#define MAX_COLUMN_PRINT 4

/* Array of grey shades */
const int grey_shades[] = {254, 251, 249, 245, 243, 239, 237, 236,
                           235, 234, 233, 232, 231, 230, 229, 228,
                           227, 226, 225, 224, 223, 222, 221};

/*******************************/
/*     Matrix Market Format    */
/*******************************/

/**
 * @brief Read a Matrix Market file and return a DoubleMatrix
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
    // printf("%zu %zu %lf\n", row_idx, col_idx, val);

    if (val != 0.0) {
      mat->row_indices[i] = (size_t)(row_idx - 1);
      mat->col_indices[i] = (size_t)(col_idx - 1);
      mat->values[i] = (double)val;
      mat->nnz++;
      // dm_set(mat, row_idx - 1, col_idx - 1, val);
    }
  }

  if (nnz > 500) {
    printf("\n");
  }

  // Close the file
  fclose(fp);

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

/**
 * @brief function print basic matrix information
 *
 * @param DoubleMatrix* mat
 */
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

void dm_print_structure(DoubleMatrix *mat, double strength) {

  // set up grid
  init_grid();

  double density = dm_density(mat);
  // information about the matrix
  printf("Structure of the matrix:\n");
  printf("Matrix (%zu x %zu, %zu), density: %lf\n", mat->rows, mat->cols,
         mat->nnz, density);

  // increase density for better visualization:
  density *= strength;

  // setup a small dense matrix to count the appearance of each element
  DoubleMatrix *count = dm_create_format(WIDTH, HEIGHT, DENSE);

  for (size_t i = 0; i < mat->nnz; i++) {
    // not every element is printed
    if (randomDouble() < density) {

      int x = get_x_coord(mat->row_indices[i], mat->rows);
      int y = get_y_coord(mat->col_indices[i], mat->cols);

      // track the number of elements in each cell
      dm_set(count, x, y, dm_get(count, x, y) + 1);
      plot(x, y, '*');
    }
  }
  // print the grid
  show_grid(count);
}

/*******************************/
/*        PLOT STRUCTURE       */
/*******************************/

void show_grid(DoubleMatrix *count) {

  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      // Check if the character has a color escape code
      int color = (int)dm_get(count, x, y);
      if (color > 1) {
        // get color escape code
        int grey_color = grey_shades[color];
        char escape_code[20];
        sprintf(escape_code, ANSI_COLOR_GREY_BASE, grey_color);

        printf("%s%c%s", escape_code, grid[y][x], ANSI_COLOR_RESET);
      } else {
        printf("%c%s", grid[y][x], ANSI_COLOR_RESET);
      }
    }
    putchar('\n');
  }
}

void init_grid(void) {
  /* Initialize grid */
  int x = 0;
  int y = 0;
  for (y = 0; y < HEIGHT; y++) {
    for (x = 0; x < WIDTH; x++) {
      grid[y][x] = ' ';
    }
  }

  /* draw the axis */
  for (y = 0; y < HEIGHT; y++) {
    grid[y][X - 1] = '|';
  }
  for (y = 0; y < HEIGHT; y++) {
    grid[y][WIDTH - 1] = '|';
  }
  for (x = 0; x < WIDTH; x++) {
    grid[Y - 1][x] = '-';
  }
  for (x = 0; x < WIDTH; x++) {
    grid[HEIGHT - 1][x] = '-';
  }

  /* set corners */
  grid[Y - 1][X - 1] = '+';
  grid[Y - 1][WIDTH - 1] = '+';
  grid[HEIGHT - 1][X - 1] = '+';
  grid[HEIGHT - 1][WIDTH - 1] = '+';
}

/*******************************/
/*       Plot Functions        */
/*******************************/

int plot(int x, int y, char c) {
  if (x > XMAX || x < XMIN || y > YMAX || y < YMIN)
    return (-1);

  grid[y][x] = c;

  return 1;
}

int get_x_coord(size_t x, size_t rows) {
  return 1 + (int)round((double)x / (double)rows * (double)(WIDTH - 2));
}

int get_y_coord(size_t y, size_t cols) {

  return 1 + (int)round((double)y / (double)cols * (double)(HEIGHT - 3));
}

/*******************************/
/* private functions (helper)  */
/*******************************/

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

static void print_progress_bar(size_t progress, size_t total, int barWidth) {
  float percentage = (float)progress / total;
  int filledWidth = (int)(percentage * barWidth);

  printf("[");
  for (int i = 0; i < barWidth; i++) {
    if (i < filledWidth) {
      printf("=");
    } else {
      printf(" ");
    }
  }
  printf("] %d%%\r", (int)(percentage * 100));
  fflush(stdout);
}
