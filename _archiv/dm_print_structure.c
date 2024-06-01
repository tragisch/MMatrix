/**
 * @file dm_print_structure.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm.h"
#include "dm_internals.h"
#include "dm_io.h"
#include "dm_math.h"

// #define BRAILLE_SIZE 10
// enum { INIT_CAPACITY = 1000U };

/* Array of grey shades */
const int grey_shades[] = {254, 251, 249, 245, 243, 239, 237, 236,
                           235, 234, 233, 232, 231, 230, 229, 228,
                           227, 226, 225, 224, 223, 222, 221};

/*******************************/
/*        STRUCTURE PLOT       */
/*******************************/

/**
 * @brief Prints the structure of a matrix to the console.
 *
 * @param mat
 * @param strength
 */
void dm_print_structure(DoubleMatrix *mat, double strength) {
  init_grid();

  double density = dm_density(mat) * strength;

  print_matrix_info(mat, density);

  DoubleMatrix *count = dm_create_format(WIDTH, HEIGHT, DENSE);

  switch (mat->format) {
  case DENSE:
    print_structure_dense(mat, count, density);
    break;
  case SPARSE:
    print_structure_coo(mat, count, density);
    break;
  default:
    break;
  }

  show_grid(count);
}

static void print_matrix_info(DoubleMatrix *mat, double density) {
  printf("Structure of the matrix:\n");
  printf("Matrix (%zu x %zu, %zu), density: %lf\n", mat->rows, mat->cols,
         mat->nnz, density);
}

static void print_element(DoubleMatrix *mat, DoubleMatrix *count, size_t x,
                          size_t y) {
  dm_set(count, x, y, dm_get(count, x, y) + 1);
  plot(x, y, '*');
}

static void print_structure_dense(DoubleMatrix *mat, DoubleMatrix *count,
                                  double density) {
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      if (mat->values[i * mat->cols + j] != 0 && randomDouble() < density) {
        int x = get_x_coord(i, mat->rows);
        int y = get_y_coord(mat->col_indices[j], mat->cols);
        print_element(mat, count, x, y);
      }
    }
  }
}

static void print_structure_coo(DoubleMatrix *mat, DoubleMatrix *count,
                                double density) {
  for (size_t i = 0; i < mat->nnz; i++) {
    if (randomDouble() < density) {
      int x = get_x_coord(mat->row_indices[i], mat->rows);
      int y = get_y_coord(mat->col_indices[i], mat->cols);
      print_element(mat, count, x, y);
    }
  }
}


/*******************************/
/*          GRID PLOT          */
/*******************************/
// from: https://c-for-dummies.com/blog/?p=761

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
    grid[y][X_DM - 1] = '|';
  }
  for (y = 0; y < HEIGHT; y++) {
    grid[y][WIDTH - 1] = '|';
  }
  for (x = 0; x < WIDTH; x++) {
    grid[Y_DM - 1][x] = '-';
  }
  for (x = 0; x < WIDTH; x++) {
    grid[HEIGHT - 1][x] = '-';
  }

  /* set corners */
  grid[Y_DM - 1][X_DM - 1] = '+';
  grid[Y_DM - 1][WIDTH - 1] = '+';
  grid[HEIGHT - 1][X_DM - 1] = '+';
  grid[HEIGHT - 1][WIDTH - 1] = '+';
}

/*******************************/
/*       Plot Functions        */
/*******************************/

int plot(int x, int y, char c) {
  if (x > XMAX || x < XMIN || y > YMAX || y < YMIN) {
    return (-1);
  }

  grid[y][x] = c;

  return 1;
}

/*******************************/
/*     Normalize to Grid       */
/*******************************/

int get_x_coord(size_t x, size_t rows) {
  return 1 + (int)round((double)x / (double)rows * (double)(WIDTH - 2));
}

int get_y_coord(size_t y, size_t cols) {

  return 1 + (int)round((double)y / (double)cols * (double)(HEIGHT - 3));
}
