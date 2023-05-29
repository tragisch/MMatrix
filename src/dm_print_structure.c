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

void dm_print_structure(DoubleMatrix *mat, double strength) {
  if (mat->format == DENSE) {
    printf(
        "Matrix is not in SPARSE or HASHTABLE format, no structure to print\n");
    return;
  }
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

  // in case of sparse matrix:
  if (mat->format == SPARSE) {
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
  } else if (mat->format == HASHTABLE) {
    khash_t(entry) *hashtable = mat->hash_table;

    // Iterate over each bucket in the hash table
    for (khint_t i = 0; i < kh_end(hashtable); ++i) {
      if (kh_exist(hashtable, i)) {
        // Retrieve the key-value pair from the current bucket
        int64_t key = kh_key(hashtable, i);
        // double value = kh_value(hashtable, i);

        // Extract row and column indices from the key
        size_t row = key >> 32;
        size_t col = key & 0xFFFFFFFF;

        if (randomDouble() < density) {
          int x = get_x_coord(row, mat->rows);
          int y = get_y_coord(col, mat->cols);

          // track the number of elements in each cell
          dm_set(count, x, y, dm_get(count, x, y) + 1);
          plot(x, y, '*');
        }
      }
    }
  }

  // print the grid
  show_grid(count);
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
