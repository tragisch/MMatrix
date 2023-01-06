#ifndef DM_IO_H
#define DM_IO_H

#include "dm_matrix.h"
#include "misc.h"

/*******************************/
/*     I/O Functions           */
/*******************************/

// read & write
int read_dm_vector_from_file(DoubleVector *vec, const char *filepath);
int write_dm_vector_to_file(DoubleVector *vec, const char *filepath);

// print stdout:
void print_dm_matrix(DoubleMatrix *matrix);
void print_dm_vector(DoubleVector *vec);

#endif  // DM_IO_H
