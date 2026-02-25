/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef DM_CONVERT_H
#define DM_CONVERT_H

#include "dm.h"
#include "dms.h"
#include "sm.h"
#include "st.h"

/**************************************/
/*       Matrix/Tensor Conversion     */
/**************************************/

// Convert sparse COO double matrix to dense double matrix.
DoubleMatrix *dms_to_dm(const DoubleSparseMatrix *src);
// Convert dense double matrix to sparse COO double matrix.
DoubleSparseMatrix *dm_to_dms(const DoubleMatrix *src);
// Convert dense float matrix to dense double matrix.
DoubleMatrix *sm_to_dm(const FloatMatrix *sm);
// Convert dense double matrix to dense float matrix.
FloatMatrix *dm_to_sm(const DoubleMatrix *src);
// Convert dense float matrix to float tensor.
FloatTensor *st_from_sm(const FloatMatrix *src);
// Convert float tensor to dense float matrix.
FloatMatrix *sm_from_st(const FloatTensor *src);

#endif // DM_CONVERT_H
