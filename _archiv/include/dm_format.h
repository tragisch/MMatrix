#ifndef DM_FORMAT_H
#define DM_FORMAT_H

#include "dm.h"

/*******************************/
/*     Matrix Market Format    */
/*******************************/

DoubleMatrix *dm_read_matrix_market(const char *filename);
void dm_write_matrix_market(const DoubleMatrix *mat, const char *filename);
void dm_read_matrix_market2(const char *filename);

#endif // DM_FORMAT_H