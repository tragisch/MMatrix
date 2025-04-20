/**
 * @file dm_cplot.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm_io.h"

/* Array of grey shades */
const int grey_shades[] = {254, 251, 249, 245, 243, 239, 237, 236,
                           235, 234, 233, 232, 231, 230, 229, 228,
                           227, 226, 225, 224, 223, 222, 221};
char grid[HEIGHT][WIDTH];  // Actual definition (only once)

/*******************************/
/*        STRUCTURE PLOT       */
/*******************************/

void dms_cplot(DoubleSparseMatrix *mat, double strength) {
  init_grid();

  double density = dms_density(mat) * strength;

  printf("Matrix (%zu x %zu, %zu), density: %lf\n", mat->rows, mat->cols,
         mat->nnz, density);
  DoubleMatrix *count = dm_create(WIDTH, HEIGHT);
  print_structure_coo(mat, count, density);

  show_grid(count);
}

void dm_cplot(DoubleMatrix *mat) {
  init_grid();

  double density = dm_density(mat);

  printf("Matrix (%zu x %zu), density: %lf\n", mat->rows, mat->cols, density);
  DoubleMatrix *count = dm_create(WIDTH, HEIGHT);
  print_structure_dense(mat, count);

  show_grid(count);
}

static void __print_element(DoubleMatrix *count, size_t x, size_t y) {
  dm_set(count, x, y, dm_get(count, x, y) + 1);
  plot(x, y, '*');
}

static void print_structure_dense(DoubleMatrix *mat, DoubleMatrix *count) {
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      if (fabs(mat->values[i * mat->cols + j]) > EPSILON) {
        int x = (int)get_x_coord(i, mat->rows);
        int y = (int)get_y_coord(j, mat->cols);
        __print_element(count, x, y);
      }
    }
  }
}

static void print_structure_coo(DoubleSparseMatrix *mat, DoubleMatrix *count,
                                double density) {
  for (size_t i = 0; i < mat->nnz; i++) {
    if (dm_rand_number() < density) {
      int x = get_x_coord(mat->row_indices[i], mat->rows);
      int y = get_y_coord(mat->col_indices[i], mat->cols);
      __print_element(count, x, y);
    }
  }
}

/*******************************/
/*          GRID PLOT          */
/*******************************/
// from: https://c-for-dummies.com/blog/?p=761

static void show_grid(DoubleMatrix *count) {
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

static void init_grid(void) {
  /* Initialize grid */
  int x = 0;
  int y = 0;
  for (y = 0; y < HEIGHT; y++) {
    for (x = 0; x < WIDTH; x++) {
      grid[y][x] = ' ';
    }
  }

  int SHIFT = 1;

  /* draw the axis */
  for (y = 0; y < HEIGHT; y++) {
    grid[y][X_DM - SHIFT] = '|';
  }
  for (y = 0; y < HEIGHT; y++) {
    grid[y][WIDTH - SHIFT] = '|';
  }
  for (x = 0; x < WIDTH; x++) {
    grid[Y_DM - SHIFT][x] = '-';
  }
  for (x = 0; x < WIDTH; x++) {
    grid[HEIGHT - SHIFT][x] = '-';
  }

  /* set corners */
  grid[Y_DM - SHIFT][X_DM - SHIFT] = '+';
  grid[Y_DM - SHIFT][WIDTH - SHIFT] = '+';
  grid[HEIGHT - SHIFT][X_DM - SHIFT] = '+';
  grid[HEIGHT - SHIFT][WIDTH - SHIFT] = '+';
}

/*******************************/
/*       Plot Functions        */
/*******************************/

static int plot(int x, int y, char c) {
  if (x > XMAX || x < XMIN || y > YMAX || y < YMIN) {
    return (-1);
  }

  grid[y][x] = c;

  return 1;
}

/*******************************/
/*     Normalize to Grid       */
/*******************************/

static int get_x_coord(size_t x, size_t rows) {
  // int ret = 1 + (int)round((double)x / (double)rows * (double)(WIDTH - 2));
  double ret = (double)(x + 1) * ((double)WIDTH / (double)(rows + 1)) - 1;
  return (int)ret;
}

static int get_y_coord(size_t y, size_t cols) {
  // int ret = 1 + (int)round((double)y / (double)cols * (double)(HEIGHT - 3));
  double ret = (double)(y) * ((double)HEIGHT / (double)(cols + 1)) + 1;
  return (int)ret;
}

/*******************************/
/*          MAT-FILEs          */
/*******************************/

int dm_write_MAT_file(const DoubleMatrix *matrix, const char *filename) {
  // Create a MAT file
  mat_t *matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);
  if (NULL == matfp) {
    fprintf(stderr, "Error creating MAT file %s\n", filename);
    return -1;
  }

  // Define the dimensions of the matrix
  size_t dims[2] = {matrix->rows, matrix->cols};

  // Create a Matio matrix variable
  matvar_t *matvar = Mat_VarCreate("matrix", MAT_C_DOUBLE, MAT_T_DOUBLE, 2,
                                   dims, matrix->values, 0);
  if (NULL == matvar) {
    fprintf(stderr, "Error creating matrix variable\n");
    Mat_Close(matfp);
    return -1;
  }

  // Write the variable to the MAT file
  Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);

  // Free the matrix variable and close the MAT file
  Mat_VarFree(matvar);
  Mat_Close(matfp);

  return 0;
}

DoubleMatrix *dm_read_MAT_file(const char *filename, const char *varname) {
  // Open the MAT file
  mat_t *matfp = Mat_Open(filename, MAT_ACC_RDONLY);
  if (NULL == matfp) {
    fprintf(stderr, "Error opening MAT file %s\n", filename);
    return NULL;
  }

  // Read the variable
  matvar_t *matvar = Mat_VarRead(matfp, varname);
  if (NULL == matvar) {
    fprintf(stderr, "Variable %s not found in MAT file %s\n", varname,
            filename);
    Mat_Close(matfp);
    return NULL;
  }

  // Extract matrix dimensions
  if (matvar->rank != 2) {
    fprintf(stderr, "Variable %s is not a 2D matrix\n", varname);
    Mat_VarFree(matvar);
    Mat_Close(matfp);
    return NULL;
  }

  size_t rows = matvar->dims[0];
  size_t cols = matvar->dims[1];

  // Create a DoubleMatrix and copy the data
  DoubleMatrix *matrix = dm_create(rows, cols);
  if (!matrix) {
    fprintf(stderr, "Error allocating memory for matrix\n");
    Mat_VarFree(matvar);
    Mat_Close(matfp);
    return NULL;
  }

  memcpy(matrix->values, matvar->data, rows * cols * sizeof(double));

  // Free the matvar and close the MAT file
  Mat_VarFree(matvar);
  Mat_Close(matfp);

  return matrix;
}

/*******************************/
/*       MARKET FILES          */
/*******************************/

void dms_write_matrix_market(const DoubleSparseMatrix *mat,
                             const char *filename) {
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
      fprintf(fp, "%zu %zu %lf\n", i + 1, j + 1, dms_get(mat, i, j));
    }
  }

  fclose(fp);
}

DoubleSparseMatrix *dms_read_matrix_market(const char *filename) {
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
  DoubleSparseMatrix *mat = dms_create(nrows, ncols, nnz);

  if (nnz > 500) {
    printf("Reading Matrix Market file: %s\n", filename);
  }

  // Read non-zero values
  for (size_t i = 0; i < nnz; i++) {
    if (nnz > 500) {
      __print_progress_bar(i, nnz, 50);
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

  return mat;
}

// if file to read is very large, print a progress bar:
static void __print_progress_bar(size_t progress, size_t total, int barWidth) {
  float percentage = (float)progress / (float)total;
  int filledWidth = (int)(percentage * (float)barWidth);

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