#ifndef DM_IO_H
#define DM_IO_H

#include "dm_matrix.h"
#include "misc.h"

/*******************************/
/*     I/O Functions           */
/*******************************/

// read & write
int dv_read_from_file(DoubleVector *vec, const char *filepath);
int dv_write_to_file(DoubleVector *vec, const char *filepath);

// print stdout:
void dm_print(DoubleMatrix *matrix);
void dv_print(DoubleVector *vec);

#endif // DM_IO_H
