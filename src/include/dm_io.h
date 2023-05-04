#ifndef DM_IO_H
#define DM_IO_H

#include "dm_matrix.h"
#include "misc.h"
#include "pbPlots.h"
#include "supportLib.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/*******************************/
/*     I/O Functions           */
/*******************************/

// read & write
int dv_read_from_file(DoubleVector *vec, const char *filepath);
int dv_write_to_file(DoubleVector *vec, const char *filepath);

// print stdout:
void dm_print(const DoubleMatrix *matrix);

void sp_print(const DoubleMatrix *mat);
void sp_print_braille(const DoubleMatrix *mat);
void sp_print_condensed(DoubleMatrix *mat);
void sp_create_scatterplot(const DoubleMatrix *mat, const char *filename);

void dv_print(const DoubleVector *vec);
static void dv_print_col(const DoubleVector *vec);
static void dv_print_row(const DoubleVector *vec);
static void print_matrix_dimension(const DoubleMatrix *mat);

#endif // DM_IO_H
