/**
 * @file m_convert.h
 * @brief Public conversion API between dense and sparse matrix types.
 */

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

/**************************************/
/*      Dense/Sparse Matrix Conversion */
/**************************************/

/**
 * @brief Convert sparse COO double matrix to dense double matrix.
 * @param src Source sparse matrix.
 * @return Dense matrix with all COO entries expanded, or NULL on allocation failure.
 */
DoubleMatrix *dms_to_dm(const DoubleSparseMatrix *src);
/**
 * @brief Convert dense double matrix to sparse COO double matrix.
 * @param src Source dense matrix.
 * @return Sparse matrix with zero entries removed, or NULL on allocation failure.
 */
DoubleSparseMatrix *dm_to_dms(const DoubleMatrix *src);
/**
 * @brief Convert dense float matrix to dense double matrix.
 * @param sm Source float matrix.
 * @return Dense double matrix with precision-expanded values, or NULL on allocation failure.
 */
DoubleMatrix *sm_to_dm(const FloatMatrix *sm);
/**
 * @brief Convert dense double matrix to dense float matrix.
 * @param src Source double matrix.
 * @return Dense float matrix with precision-reduced values, or NULL on allocation failure.
 */
FloatMatrix *dm_to_sm(const DoubleMatrix *src);
#endif  // DM_CONVERT_H
