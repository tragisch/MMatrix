#ifndef DM_IO_H
#define DM_IO_H

#include <matio.h>

#include "dm.h"
#include "dms.h"

/*******************************/
/*     Plot DEFINES            */
/*******************************/

#define WIDTH (36)   // 44
#define HEIGHT (18)  // 22
#define X_DM (1)
#define Y_DM (1)
#define XMAX (WIDTH - X_DM - 1)
#define XMIN (1)  // -(WIDTH - X)
#define YMAX (HEIGHT - Y_DM - 1)
#define YMIN (1)  // -(HEIGHT - Y) + 1
#define MAX_NUMBER_OF_COLUMNS (30)
#define MAX_NUMBER_OF_ROWS (30)

/* ANSI escape codes for colors */
#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_COLOR_GREY_BASE "\x1b[48;5;%dm"

extern char grid[HEIGHT][WIDTH];

/*******************************/
/*          I/O Functions      */
/*******************************/

void dm_cplot(DoubleMatrix *mat);
void dms_cplot(DoubleSparseMatrix *mat, double strength);

int dm_write_MAT_file(const DoubleMatrix *matrix, const char *filename);
DoubleMatrix *dm_read_MAT_file(const char *filename, const char *varname);

DoubleSparseMatrix *dms_read_matrix_market(const char *filename);
void dms_write_matrix_market(const DoubleSparseMatrix *mat,
                             const char *filename);

/*******************************/
/*     Help Function Plot      */
/*******************************/

static void print_structure_coo(DoubleSparseMatrix *mat, DoubleMatrix *count,
                                double density);
static void print_structure_dense(DoubleMatrix *mat, DoubleMatrix *count);
static void __print_element(DoubleMatrix *count, size_t x, size_t y);
static int plot(int x, int y, char c);
static void init_grid(void);
static void show_grid(DoubleMatrix *count);
static int get_x_coord(size_t x, size_t rows);
static int get_y_coord(size_t y, size_t cols);
static void __print_progress_bar(size_t progress, size_t total, int barWidth);

#endif  // DM_IO_H
