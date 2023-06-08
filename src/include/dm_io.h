#ifndef DM_IO_H
#define DM_IO_H

#include "dm.h"
#include "dm_internals.h"
#include <cc_io.h>

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
/*     Pretty Print Matrix      */
/*******************************/

// print stdout:
void dm_print(const DoubleMatrix *matrix);

// test braille
void sp_print_braille(const DoubleMatrix *mat);

/*******************************/
/*   Print brief information   */
/*******************************/

void dm_brief(const DoubleMatrix *mat);
static void dm_brief_sparse(const DoubleMatrix *mat);

void dm_print_condensed(DoubleMatrix *mat);

/*******************************/
/*         Print Vector        */
/*******************************/

void dv_print(const DoubleVector *vec);
static void dv_print_col(const DoubleVector *vec);
static void dv_print_row(const DoubleVector *vec);
static void print_matrix_dimension(const DoubleMatrix *mat);

// test
void print_ccs(DoubleMatrix *mat);

/*******************************/
/*        STRUCTURE PLOT       */
/*******************************/

void dm_print_structure(DoubleMatrix *mat, double strength);

static void print_structure_csc(DoubleMatrix *mat, DoubleMatrix *count,
                                double density);
static void print_structure_coo(DoubleMatrix *mat, DoubleMatrix *count,
                                double density);
static void print_structure_dense(DoubleMatrix *mat, DoubleMatrix *count,
                                  double density);
static void print_element(DoubleMatrix *mat, DoubleMatrix *count, size_t x,
                          size_t y);
static void print_matrix_dimension(const DoubleMatrix *mat);
static void print_matrix_info(DoubleMatrix *mat, double density);

/*******************************/
/*     Help Function Plot      */
/*******************************/

int plot(int x, int y, char c);

void init_grid(void);
void show_grid(DoubleMatrix *count);
int get_x_coord(size_t x, size_t rows);
int get_y_coord(size_t y, size_t cols);

#endif // DM_IO_H
