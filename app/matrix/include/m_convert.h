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

DoubleMatrix *dms_to_dm(const DoubleSparseMatrix *dms_create_empty);
DoubleSparseMatrix *dm_to_dms(const DoubleMatrix *dm_create_empty);
DoubleMatrix *sm_to_dm(const FloatMatrix *sm);
FloatMatrix *dm_to_sm(const DoubleMatrix *dm_create_empty);

#endif // DM_CONVERT_H
