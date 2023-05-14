#include "dm_io.h"

#include "dbg.h"
#include <math.h>

/*******************************/
/*          Intials            */
/*******************************/

void show_grid(void) {
  int x = 0;
  int y = 0;
  for (y = 0; y < HEIGHT; y++) {
    for (x = 0; x < WIDTH; x++) {
      putchar(grid[y][x]);
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

  if (x > XMAX || x < XMIN || y > YMAX || y < YMIN) {
    return (-1);
  }
  grid[y][x] = c; // '*';
  return (1);
}

int get_x_coord(size_t x, size_t rows) {
  return (int)round((double)x / (double)rows * (double)WIDTH);
}

int get_y_coord(size_t y, size_t cols) {
  return (int)round((double)y / (double)cols * (double)HEIGHT);
}
