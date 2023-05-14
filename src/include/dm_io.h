#ifndef DM_IO_H
#define DM_IO_H

#include "dm_matrix.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/*******************************/
/*     I/O Functions           */
/*******************************/

// read & write
DoubleMatrix *dm_read_matrix_market(const char *filename);

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

#endif // DM_IO_H
