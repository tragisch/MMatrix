/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#include "m_io.h"

#include <log.h>
#include <math.h>

#include "sm.h"

#ifndef INIT_CAPACITY
#define INIT_CAPACITY 100
#endif
#ifndef EPSILON
#define EPSILON 1e-9
#endif

/* Array of heatmap-inspired ANSI 256-color codes */
const int grey_shades[] = {
    21,  27,  33,  39,  45,  51, 50, 49, 48, 47,  // blue to cyan to green
    46,  82,  118, 154, 190,                      // green to yellow
    226, 220, 214, 208, 202, 196                  // yellow to red
};
char grid[HEIGHT][WIDTH];  // Actual definition (only once)

/*******************************/
/*       Plot & Progress       */
/*******************************/

static int plot(int x, int y, char c) {
  if (x > XMAX || x < XMIN || y > YMAX || y < YMIN) {
    return (-1);
  }

  grid[y][x] = c;

  return 1;
}

const char *m_status_to_string(MStatus status) {
  switch (status) {
    case MSTATUS_OK:
      return "OK";
    case MSTATUS_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case MSTATUS_IO_ERROR:
      return "IO_ERROR";
    case MSTATUS_ALLOC_FAILED:
      return "ALLOC_FAILED";
    case MSTATUS_FORMAT_ERROR:
      return "FORMAT_ERROR";
    case MSTATUS_UNSUPPORTED_TYPE:
      return "UNSUPPORTED_TYPE";
    case MSTATUS_INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    default:
      return "UNKNOWN";
  }
}

// if file to read is very large, print a progress bar:
static void print_progress_bar(size_t progress, size_t totalSteps,
                               int barWidth) {
  if (totalSteps == 0) return;
  float percentage = (float)progress / (float)totalSteps;
  int filledWidth = (int)(percentage * (float)barWidth);

  printf("[");
  for (int i = 0; i < barWidth; i++) {
    if (i < filledWidth) {
      printf("=");
    } else {
      printf(" ");
    }
  }
  printf("] %3d%%\r", (int)(percentage * 100));
  fflush(stdout);
}

/*******************************/
/*     Normalize to Grid       */
/*******************************/

static int get_cols_coord(size_t x, size_t rows) {
  // int ret = 1 + (int)round((double)x / (double)rows * (double)(WIDTH - 2));
  double ret = (double)(x + 1) * ((double)WIDTH / (double)(rows + 1));
  return (int)ret;
}

static int get_rows_coord(size_t y, size_t cols) {
  // int ret = 1 + (int)round((double)y / (double)cols * (double)(HEIGHT - 3));
  double ret = 1 + (double)(y) * ((double)HEIGHT / (double)(cols + 1)) - 1;
  return (int)ret;
}

/*******************************/
/*          GRID PLOT          */
/*******************************/
// from: https://c-for-dummies.com/blog/?p=761

static void show_grid(FloatMatrix *count) {
  int n_shades = sizeof(grey_shades) / sizeof(grey_shades[0]);
  float max_count = 0.0f;
  for (size_t y = 0; y < count->rows; y++) {
    for (size_t x = 0; x < count->cols; x++) {
      float val = sm_get(count, x, y);
      if (val > max_count) {
        max_count = val;
      }
    }
  }
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      // Check if the character has a color escape code
      int color = (int)sm_get(count, (size_t)x, (size_t)y);
      if (color >= 1) {
        int shade_index =
            (int)((log1pf((float)color) / log1pf((float)max_count)) *
                  (int)(n_shades - 1));
        if (shade_index >= n_shades) shade_index = n_shades - 1;
        int grey_color = grey_shades[shade_index];
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
/*        STRUCTURE PLOT       */
/*******************************/

static void print_element(FloatMatrix *count, size_t x, size_t y,
                          double density) {
  // use different character for density
  char c;
  if (density < 0.02) {
    c = '*';
  } else if (density < 0.05 && density >= 0.02) {
    c = '.';
  } else if (density >= 0.05) {
    c = ' ';
  } else {
    c = '#';
  }
  if (x < count->cols && y < count->rows) {
    count->values[y * count->cols + x]++;
    plot((int)y, (int)x, c);
  }
}

static void print_structure_coo(DoubleSparseMatrix *mat, FloatMatrix *count,
                                double density) {
  for (size_t i = 0; i < mat->nnz; i++) {
#ifdef __APPLE__
    uint32_t random_uint32 = arc4random();
#else
    uint32_t random_uint32 = (uint32_t)rand();
#endif
    double rand_number = (double)random_uint32 / (double)UINT32_MAX;

    if (rand_number < density) {
      int x = get_cols_coord(mat->col_indices[i], mat->cols);
      int y = get_rows_coord(mat->row_indices[i], mat->rows);
      print_element(count, (size_t)y, (size_t)x, density);
    }
  }
}

void dms_cplot(DoubleSparseMatrix *mat, double strength) {
  init_grid();

  double density = dms_density(mat) * strength;

  printf("Matrix (%zu x %zu, %zu), density: %lf\n", mat->rows, mat->cols,
         mat->nnz, density);
  FloatMatrix *count = sm_create(WIDTH, HEIGHT);
  print_structure_coo(mat, count, density);

  show_grid(count);
}

void dm_cplot(DoubleMatrix *mat) {
  init_grid();

  double density = dm_density(mat);

  printf("Matrix (%zu x %zu), density: %lf\n", mat->rows, mat->cols, density);
  FloatMatrix *count = sm_create(WIDTH, HEIGHT);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      if (fabs(mat->values[i * mat->cols + j]) > EPSILON) {
        int x = (int)get_cols_coord(i, mat->rows);
        int y = (int)get_rows_coord(j, mat->cols);
        print_element(count, (size_t)y, (size_t)x, 1.0);
      }
    }
  }

  show_grid(count);
}

void sm_cplot(FloatMatrix *mat) {
  init_grid();
  float density = sm_density(mat);
  printf("Matrix (%zu x %zu), density: %f\n", mat->rows, mat->cols, density);
  FloatMatrix *count = sm_create(WIDTH, HEIGHT);

  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      if (fabsf(mat->values[i * mat->cols + j]) > EPSILON) {
        int x = (int)get_cols_coord(j, mat->rows);
        int y = (int)get_rows_coord(i, mat->cols);
        print_element(count, (size_t)y, (size_t)x, 1.0);
      }
    }
  }

  show_grid(count);
}

/*******************************/
/*          MAT-FILEs          */
/*******************************/

void mio_set_format(MIOFormat fmt) { g_mio_format = fmt; }
MIOFormat mio_get_format(void) { return g_mio_format; }

void mio_set_compression(MIOCompression comp) { g_mio_compression = comp; }
MIOCompression mio_get_compression(void) { return g_mio_compression; }

static MStatus write_MAT_file_generic(const char *filename, size_t rows,
                                      size_t cols, void *data,
                                      enum matio_classes cls,
                                      enum matio_types type,
                                      enum mat_ft version,
                                      enum matio_compression compression) {
  if (!filename || !data) {
    return MSTATUS_INVALID_ARGUMENT;
  }
  if (cols != 0 && rows > SIZE_MAX / cols) {
    log_error("Invalid matrix dimensions for MAT write (%zu x %zu)", rows,
              cols);
    return MSTATUS_INVALID_ARGUMENT;
  }

  size_t elements = rows * cols;
  void *column_major_data = NULL;

  if (type == MAT_T_DOUBLE) {
    double *src = (double *)data;
    double *dst = (double *)malloc(elements * sizeof(double));
    if (!dst) {
      log_error("Failed to allocate conversion buffer for MAT write");
      return MSTATUS_ALLOC_FAILED;
    }

    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        dst[col * rows + row] = src[row * cols + col];
      }
    }
    column_major_data = dst;
  } else if (type == MAT_T_SINGLE) {
    float *src = (float *)data;
    float *dst = (float *)malloc(elements * sizeof(float));
    if (!dst) {
      log_error("Failed to allocate conversion buffer for MAT write");
      return MSTATUS_ALLOC_FAILED;
    }

    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        dst[col * rows + row] = src[row * cols + col];
      }
    }
    column_major_data = dst;
  } else {
    column_major_data = data;
  }

  mat_t *matfp = Mat_CreateVer(filename, NULL, version);
  if (!matfp) {
    log_error("Fehler beim Erstellen der MAT-Datei %s", filename);
    if (column_major_data != data) {
      free(column_major_data);
    }
    return MSTATUS_IO_ERROR;
  }

  size_t dims[2] = {rows, cols};
  matvar_t *matvar =
      Mat_VarCreate("matrix", cls, type, 2, dims, column_major_data, 0);
  if (!matvar) {
    log_error("Fehler beim Erstellen der Matrix-Variable");
    Mat_Close(matfp);
    if (column_major_data != data) {
      free(column_major_data);
    }
    return MSTATUS_ALLOC_FAILED;
  }

  Mat_VarWrite(matfp, matvar, compression);
  Mat_VarFree(matvar);
  Mat_Close(matfp);

  if (column_major_data != data) {
    free(column_major_data);
  }

  return MSTATUS_OK;
}

MStatus dm_write_mat_file_ex(const DoubleMatrix *matrix, const char *filename) {
  if (!matrix || !matrix->values) {
    return MSTATUS_INVALID_ARGUMENT;
  }
  enum matio_compression comp = (g_mio_compression == MIO_COMPRESS_ZLIB)
                                    ? MAT_COMPRESSION_ZLIB
                                    : MAT_COMPRESSION_NONE;

  return write_MAT_file_generic(filename, matrix->rows, matrix->cols,
                                matrix->values, MAT_C_DOUBLE, MAT_T_DOUBLE,
                                MAT_FT_MAT5, comp);
}

MStatus sm_write_mat_file_ex(const FloatMatrix *matrix, const char *filename) {
  if (!matrix || !matrix->values) {
    return MSTATUS_INVALID_ARGUMENT;
  }
  enum matio_compression comp = (g_mio_compression == MIO_COMPRESS_ZLIB)
                                    ? MAT_COMPRESSION_ZLIB
                                    : MAT_COMPRESSION_NONE;

  return write_MAT_file_generic(filename, matrix->rows, matrix->cols,
                                matrix->values, MAT_C_SINGLE, MAT_T_SINGLE,
                                MAT_FT_MAT5, comp);
}

MStatus dm_write_MAT_file_ex(const DoubleMatrix *matrix, const char *filename) {
  return dm_write_mat_file_ex(matrix, filename);
}

MStatus sm_write_MAT_file_ex(const FloatMatrix *matrix, const char *filename) {
  return sm_write_mat_file_ex(matrix, filename);
}

int dm_write_mat_file(const DoubleMatrix *matrix, const char *filename) {
  return dm_write_mat_file_ex(matrix, filename) == MSTATUS_OK ? 0 : -1;
}

int sm_write_mat_file(const FloatMatrix *matrix, const char *filename) {
  return sm_write_mat_file_ex(matrix, filename) == MSTATUS_OK ? 0 : -1;
}

int dm_write_MAT_file(const DoubleMatrix *matrix, const char *filename) {
  return dm_write_mat_file(matrix, filename);
}

int sm_write_MAT_file(const FloatMatrix *matrix, const char *filename) {
  return sm_write_mat_file(matrix, filename);
}

typedef void *(*matrix_alloc_fn)(size_t rows, size_t cols);

static MStatus read_MAT_variable(const char *filename, matvar_t **out_matvar) {
  if (!filename || !out_matvar) {
    return MSTATUS_INVALID_ARGUMENT;
  }

  *out_matvar = NULL;
  mat_t *matfp = Mat_Open(filename, MAT_ACC_RDONLY);

  if (!matfp) {
    log_error("Error opening MAT file: %s", filename);
    return MSTATUS_IO_ERROR;
  }

  // loop through all variables in the MAT file

  matvar_t *matvar;
  int var_count = 0;
  char varname[256] = {0};

  while ((matvar = Mat_VarReadNext(matfp)) != NULL) {
    var_count++;
    if (var_count == 1) {
      // Name of the first variable
      strncpy(varname, matvar->name, sizeof(varname) - 1);
    }
    Mat_VarFree(matvar);
  }

  if (var_count == 0) {
    log_error("No variables found in MAT file.\n");
    Mat_Close(matfp);
    return MSTATUS_FORMAT_ERROR;
  }

  if (var_count > 1) {
    log_error("More than one variable found (%d). Not suppoted yet.\n",
              var_count);
    Mat_Close(matfp);
    return MSTATUS_FORMAT_ERROR;
  }

  matvar = Mat_VarRead(matfp, varname);
  if (!matvar) {
    log_error("Failed to read variable '%s'\n", varname);
    Mat_Close(matfp);
    return MSTATUS_IO_ERROR;
  }

  if (matvar->rank != 2) {
    log_error("Variable %s is not a 2D-Matrix", varname);
    Mat_VarFree(matvar);
    Mat_Close(matfp);
    return MSTATUS_FORMAT_ERROR;
  }

  Mat_Close(matfp);
  *out_matvar = matvar;
  return MSTATUS_OK;
}

MStatus dm_read_mat_file_ex(const char *filename, DoubleMatrix **out_matrix) {
  if (!out_matrix) {
    return MSTATUS_INVALID_ARGUMENT;
  }
  *out_matrix = NULL;

  matvar_t *matvar = NULL;
  MStatus st = read_MAT_variable(filename, &matvar);
  if (st != MSTATUS_OK) return st;

  if (matvar->class_type == MAT_C_SPARSE) {
    log_error("Expected dense matrix, but sparse matrix found in file %s",
              filename);
    Mat_VarFree(matvar);
    return MSTATUS_FORMAT_ERROR;
  }

  size_t rows = matvar->dims[0];
  size_t cols = matvar->dims[1];
  DoubleMatrix *matrix = dm_create(rows, cols);
  if (!matrix) {
    log_error("Error allocating matrix\n");
    Mat_VarFree(matvar);
    return MSTATUS_ALLOC_FAILED;
  }
  if (matvar->data_type == MAT_T_SINGLE) {
    float *data = (float *)matvar->data;
    for (size_t col = 0; col < cols; ++col) {
      for (size_t row = 0; row < rows; ++row) {
        matrix->values[row * cols + col] = (double)data[col * rows + row];
      }
    }
  } else if (matvar->data_type == MAT_T_DOUBLE) {
    double *data = (double *)matvar->data;
    for (size_t col = 0; col < cols; ++col) {
      for (size_t row = 0; row < rows; ++row) {
        matrix->values[row * cols + col] = data[col * rows + row];
      }
    }
  } else {
    log_error("Unsupported data type: %d\n", matvar->data_type);
    dm_destroy(matrix);
    Mat_VarFree(matvar);
    return MSTATUS_UNSUPPORTED_TYPE;
  }
  Mat_VarFree(matvar);
  *out_matrix = matrix;
  return MSTATUS_OK;
}

MStatus sm_read_mat_file_ex(const char *filename, FloatMatrix **out_matrix) {
  if (!out_matrix) {
    return MSTATUS_INVALID_ARGUMENT;
  }
  *out_matrix = NULL;

  matvar_t *matvar = NULL;
  MStatus st = read_MAT_variable(filename, &matvar);
  if (st != MSTATUS_OK) {
    return st;
  }

  if (matvar->class_type == MAT_C_SPARSE) {
    log_error("Expected dense matrix, but sparse matrix found in file %s",
              filename);
    Mat_VarFree(matvar);
    return MSTATUS_FORMAT_ERROR;
  }

  size_t rows = matvar->dims[0];
  size_t cols = matvar->dims[1];
  FloatMatrix *matrix = sm_create(rows, cols);
  if (!matrix) {
    log_error("Error allocating matrix\n");
    Mat_VarFree(matvar);
    return MSTATUS_ALLOC_FAILED;
  }
  if (matvar->data_type == MAT_T_SINGLE) {
    float *data = (float *)matvar->data;
    for (size_t col = 0; col < cols; ++col) {
      for (size_t row = 0; row < rows; ++row) {
        matrix->values[row * cols + col] = data[col * rows + row];
      }
    }
  } else if (matvar->data_type == MAT_T_DOUBLE) {
    double *data = (double *)matvar->data;
    for (size_t col = 0; col < cols; ++col) {
      for (size_t row = 0; row < rows; ++row) {
        matrix->values[row * cols + col] = (float)data[col * rows + row];
      }
    }
  } else {
    log_error("Unsupported data type: %d\n", matvar->data_type);
    sm_destroy(matrix);
    Mat_VarFree(matvar);
    return MSTATUS_UNSUPPORTED_TYPE;
  }
  Mat_VarFree(matvar);
  *out_matrix = matrix;
  return MSTATUS_OK;
}

MStatus dms_read_mat_file_ex(const char *filename,
                             DoubleSparseMatrix **out_matrix) {
  if (!out_matrix) {
    return MSTATUS_INVALID_ARGUMENT;
  }
  *out_matrix = NULL;

  matvar_t *matvar = NULL;
  MStatus st = read_MAT_variable(filename, &matvar);
  if (st != MSTATUS_OK) {
    return st;
  }

  if (!(matvar->class_type == MAT_C_SPARSE)) {
    log_error("Expected sparse matrix, but dense matrix found in file %s",
              filename);
    Mat_VarFree(matvar);
    return MSTATUS_FORMAT_ERROR;
  }

  mat_sparse_t *s = (mat_sparse_t *)matvar->data;
  size_t rows = matvar->dims[0];
  size_t cols = matvar->dims[1];
  size_t nnz = s->nzmax;

  DoubleSparseMatrix *mat = dms_create(rows, cols, nnz);
  if (!mat) {
    log_error("Failed to allocate sparse matrix structure");
    Mat_VarFree(matvar);
    return MSTATUS_ALLOC_FAILED;
  }

  size_t count = 0;
  for (size_t col = 0; col < cols; ++col) {
    for (size_t k = (size_t)s->jc[col]; k < (size_t)s->jc[col + 1]; ++k) {
      if (count >= mat->capacity) {
        if (mat->capacity == 0) {
          mat->capacity = 1;
        } else {
          mat->capacity *= 2;
        }
        void *tmp = realloc(mat->row_indices, mat->capacity * sizeof(size_t));
        if (!tmp) {
          log_error("Realloc failed");
          dms_destroy(mat);
          Mat_VarFree(matvar);
          return MSTATUS_ALLOC_FAILED;
        }
        mat->row_indices = tmp;
        tmp = realloc(mat->col_indices, mat->capacity * sizeof(size_t));
        if (!tmp) {
          log_error("Realloc failed");
          dms_destroy(mat);
          Mat_VarFree(matvar);
          return MSTATUS_ALLOC_FAILED;
        }
        mat->col_indices = tmp;
        tmp = realloc(mat->values, mat->capacity * sizeof(double));
        if (!tmp) {
          log_error("Realloc failed");
          dms_destroy(mat);
          Mat_VarFree(matvar);
          return MSTATUS_ALLOC_FAILED;
        }
        mat->values = tmp;
      }
      mat->row_indices[count] = (size_t)s->ir[k];
      mat->col_indices[count] = col;
      mat->values[count] = ((double *)s->data)[k];
      count++;
    }
  }

  mat->nnz = count;
  Mat_VarFree(matvar);
  *out_matrix = mat;
  return MSTATUS_OK;
}

MStatus dm_read_MAT_file_ex(const char *filename, DoubleMatrix **out_matrix) {
  return dm_read_mat_file_ex(filename, out_matrix);
}

MStatus sm_read_MAT_file_ex(const char *filename, FloatMatrix **out_matrix) {
  return sm_read_mat_file_ex(filename, out_matrix);
}

MStatus dms_read_MAT_file_ex(const char *filename,
                             DoubleSparseMatrix **out_matrix) {
  return dms_read_mat_file_ex(filename, out_matrix);
}

DoubleMatrix *dm_read_mat_file(const char *filename) {
  DoubleMatrix *matrix = NULL;
  if (dm_read_mat_file_ex(filename, &matrix) != MSTATUS_OK) {
    return NULL;
  }
  return matrix;
}

FloatMatrix *sm_read_mat_file(const char *filename) {
  FloatMatrix *matrix = NULL;
  if (sm_read_mat_file_ex(filename, &matrix) != MSTATUS_OK) {
    return NULL;
  }
  return matrix;
}

DoubleSparseMatrix *dms_read_mat_file(const char *filename) {
  DoubleSparseMatrix *matrix = NULL;
  if (dms_read_mat_file_ex(filename, &matrix) != MSTATUS_OK) {
    return NULL;
  }
  return matrix;
}

DoubleMatrix *dm_read_MAT_file(const char *filename) {
  return dm_read_mat_file(filename);
}

FloatMatrix *sm_read_MAT_file(const char *filename) {
  return sm_read_mat_file(filename);
}

DoubleSparseMatrix *dms_read_MAT_file(const char *filename) {
  return dms_read_mat_file(filename);
}

/*******************************/
/*       MARKET FILES          */
/*******************************/

void dms_write_matrix_market(const DoubleSparseMatrix *mat,
                             const char *filename) {
  FILE *fp = NULL;
  fp = fopen(filename, "w");
  if (fp == NULL) {
    log_error("Error: Unable to open file.\n");
    exit(1);
  }

  fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(fp, "%zu %zu %zu\n", mat->rows, mat->cols, mat->nnz);

  for (size_t k = 0; k < mat->nnz; ++k) {
    fprintf(fp, "%zu %zu %.6lf\n", mat->row_indices[k] + 1,
            mat->col_indices[k] + 1, mat->values[k]);
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
    log_error("Error: Unable to open file.\n");
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
  if (sscanf(line, "%zu %zu %zu", &nrows, &ncols, &nnz) != 3) {
    log_error("Failed to read matrix dimensions.");
    fclose(fp);
    return NULL;
  }

  // Create DoubleMatrix
  DoubleSparseMatrix *mat = dms_create(nrows, ncols, nnz);

  if (nnz > 500) {
    printf("Reading Matrix Market file: %s\n", filename);
  }

  // Read non-zero values
  for (size_t i = 0; i < nnz; i++) {
    if (nnz > 500) {
      print_progress_bar(/*total_steps=*/nnz, /*current_step=*/i,
                         /*bar_width=*/50);
    }
    size_t row_idx = 0;
    size_t col_idx = 0;
    double val = 0.0;

    if (fscanf(fp, "%zu %zu %lf", &row_idx, &col_idx, &val) != 3) {
      log_error("Failed to read matrix element at line %zu", i);
      break;
    }

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
