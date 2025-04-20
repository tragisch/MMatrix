#ifndef DM_CONVERT_H
#define DM_CONVERT_H

#include "dm.h"
#include "dms.h"

DoubleMatrix *dms_to_dm(const DoubleSparseMatrix *dms);
DoubleSparseMatrix *dm_to_dms(const DoubleMatrix *dm_create_empty);

#endif // DM_CONVERT_H
