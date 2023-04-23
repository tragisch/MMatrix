#ifndef DM_IO_H
#define DM_IO_H

#include "dm_matrix.h"
#include "dv_vector.h"
#include "misc.h"
#include "pbPlots.h"
#include "sp_matrix.h"
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

void sp_print(const SparseMatrix *mat);
void sp_print_braille(const SparseMatrix *mat);
void sp_print_condensed(SparseMatrix *mat);
void sp_create_scatterplot(const SparseMatrix *mat, const char *filename);

void dv_print(const DoubleVector *vec);
static void dv_print_col(const DoubleVector *vec);
static void dv_print_row(const DoubleVector *vec);

#endif // DM_IO_H
