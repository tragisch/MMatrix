/**
 * @file m_convert.h
 * @brief Public conversion API between dense/sparse matrix and tensor types.
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
#include "st.h"

/**************************************/
/*       Matrix/Tensor Conversion     */
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
/**
 * @brief Convert dense float matrix to float tensor.
 * @param src Source float matrix (shape: rows x cols interpreted as batch 1, rows, cols, 1).
 * @return Float tensor with matrix reshaped, or NULL on allocation failure.
 */
FloatTensor *st_from_sm(const FloatMatrix *src);
/**
 * @brief Convert float tensor to dense float matrix.
 * @param src Source float tensor (extracted as 2D: flatten/reshape to rows x cols).
 * @return Dense float matrix, or NULL on allocation failure or invalid tensor shape.
 */
FloatMatrix *sm_from_st(const FloatTensor *src);

#endif  // DM_CONVERT_H
