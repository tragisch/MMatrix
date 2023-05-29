#ifndef DM_IO_H
#define DM_IO_H

#include "dm.h"
#include "dm_internals.h"

/*******************************/
/*     Plot DEFINES            */
/*******************************/

#define WIDTH (36)  // 44
#define HEIGHT (18) // 22
#define X_DM (1)
#define Y_DM (1)
#define XMAX (WIDTH - X_DM - 1)
#define XMIN (1) // -(WIDTH - X)
#define YMAX (HEIGHT - Y_DM - 1)
#define YMIN (1) // -(HEIGHT - Y) + 1
#define MAX_NUMBER_OF_COLUMNS (30)
#define MAX_NUMBER_OF_ROWS (30)

/* ANSI escape codes for colors */
#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_COLOR_GREY_BASE "\x1b[48;5;%dm"

char grid[HEIGHT][WIDTH];

/*******************************/
/*     I/O Functions           */
/*******************************/

// read & write
DoubleMatrix *dm_read_matrix_market(const char *filename);
void dm_write_matrix_market(const DoubleMatrix *mat, const char *filename);


// print stdout:
void dm_brief(const DoubleMatrix *mat);
void dm_print(const DoubleMatrix *matrix);

void dm_brief_sparse(const DoubleMatrix *mat);
void sp_print_braille(const DoubleMatrix *mat);
void dm_print_condensed(DoubleMatrix *mat);
void dm_print_structure(DoubleMatrix *mat, double strength);

void dv_print(const DoubleVector *vec);
static void dv_print_col(const DoubleVector *vec);
static void dv_print_row(const DoubleVector *vec);
static void print_matrix_dimension(const DoubleMatrix *mat);

/*******************************/
/*     Plot structure         */
/*******************************/

int plot(int x, int y, char c);

void init_grid(void);
void show_grid(DoubleMatrix *count);
int get_x_coord(size_t x, size_t rows);
int get_y_coord(size_t y, size_t cols);

#endif // DM_IO_H
