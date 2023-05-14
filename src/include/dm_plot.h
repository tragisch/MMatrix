#ifndef DM_TEXT_MODE_PLOT_H
#define DM_TEXT_MODE_PLOT_H

#include <stdio.h>

#define WIDTH 44
#define HEIGHT 22
#define X 1
#define Y 1
#define XMAX WIDTH - X - 1
#define XMIN 1 // -(WIDTH - X)
#define YMAX HEIGHT - Y - 1
#define YMIN 1 // -(HEIGHT - Y) + 1

char grid[HEIGHT][WIDTH];

void init_grid(void);
void show_grid(void);

int plot(int x, int y, char c);
int get_x_coord(size_t x, size_t rows);
int get_y_coord(size_t y, size_t cols);

#endif // DM_TEXT_MODE_PLOT_H