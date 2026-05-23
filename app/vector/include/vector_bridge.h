/**
 * @file vector_bridge.h
 * @brief Bridge helpers between vectors, matrices, sparse matrices, and tensors.
 */

#ifndef VECTOR_BRIDGE_H
#define VECTOR_BRIDGE_H

#include "dm.h"
#include "dms.h"
#include "dv.h"
#include "dvs.h"
#include "sm.h"
#include "sv.h"
#include "vv.h"

FloatVectorView sm_row_view(FloatMatrix *mat, size_t row);
FloatVectorView sm_col_view(FloatMatrix *mat, size_t col);
FloatVector *sm_row_to_sv(const FloatMatrix *mat, size_t row);
FloatVector *sm_col_to_sv(const FloatMatrix *mat, size_t col);
FloatVector *sm_matvec(const FloatMatrix *mat, const FloatVector *vec);
FloatMatrix *sv_outer_as_sm(const FloatVector *lhs, const FloatVector *rhs);

DoubleVector *dm_row_to_dv(const DoubleMatrix *mat, size_t row);
DoubleVector *dm_col_to_dv(const DoubleMatrix *mat, size_t col);
DoubleVector *dm_matvec(const DoubleMatrix *mat, const DoubleVector *vec);
DoubleMatrix *dv_outer_as_dm(const DoubleVector *lhs, const DoubleVector *rhs);

DoubleSparseVector *dms_row_to_dvs(const DoubleSparseMatrix *mat, size_t row);
DoubleSparseVector *dms_col_to_dvs(const DoubleSparseMatrix *mat, size_t col);

#endif  // VECTOR_BRIDGE_H
