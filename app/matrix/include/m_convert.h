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

DoubleMatrix *dms_to_dm(const DoubleSparseMatrix *src);
DoubleSparseMatrix *dm_to_dms(const DoubleMatrix *src);
DoubleMatrix *sm_to_dm(const FloatMatrix *sm);
FloatMatrix *dm_to_sm(const DoubleMatrix *src);
FloatTensor *st_from_sm(const FloatMatrix *src);
FloatMatrix *sm_from_st(const FloatTensor *src);

#endif // DM_CONVERT_H
