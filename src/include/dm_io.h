#ifndef DM_IO_H
#define DM_IO_H

#include "dm.h"
#include "dm_internals.h"

#define WIDTH 44
#define HEIGHT 22
#define X 1
#define Y 1
#define XMAX WIDTH - X - 1
#define XMIN 1 // -(WIDTH - X)
#define YMAX HEIGHT - Y - 1
#define YMIN 1 // -(HEIGHT - Y) + 1

/*******************************/
/*     I/O Functions           */
/*******************************/

// read & write
DoubleMatrix *dm_read_matrix_market(const char *filename);
void dm_write_matrix_market(const DoubleMatrix *mat, const char *filename);
static void printProgressBar(size_t progress, size_t total, int barWidth);

// print stdout:
void dm_brief(const DoubleMatrix *mat);
void dm_print(const DoubleMatrix *matrix);

void sp_print(const DoubleMatrix *mat);
void sp_print_braille(const DoubleMatrix *mat);
void sp_print_condensed(DoubleMatrix *mat);
void dm_print_structure(DoubleMatrix *mat);

void dv_print(const DoubleVector *vec);
static void dv_print_col(const DoubleVector *vec);
static void dv_print_row(const DoubleVector *vec);
static void print_matrix_dimension(const DoubleMatrix *mat);

/*******************************/
/*     Plot scructure         */
/*******************************/

int plot(int x, int y, char c);
char grid[HEIGHT][WIDTH];
void init_grid(void);
void show_grid(void);
int get_x_coord(size_t x, size_t rows);
int get_y_coord(size_t y, size_t cols);

#endif // DM_IO_H
