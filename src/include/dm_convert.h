#ifndef DM_CONVERT_H
#define DM_CONVERT_H

#include "dm.h"
#include "khash.h"
#include <stdbool.h>

/*******************************/
/*     Define & Types          */
/*******************************/

void dm_convert(DoubleMatrix *mat, matrix_format format);

static void dm_convert_sparse_to_dense(DoubleMatrix *mat);
static void dm_convert_dense_to_sparse(DoubleMatrix *mat);

#endif //  DM_CONVERT_H