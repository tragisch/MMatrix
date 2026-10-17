/**
 * @file st_convert.h
 * @brief Conversion helpers between tensor and matrix types.
 */

#ifndef ST_CONVERT_H
#define ST_CONVERT_H

#include "sm.h"
#include "st.h"

/**
 * @brief Convert dense float matrix to float tensor.
 * @param src Source float matrix.
 * @return Float tensor with shape [rows, cols], or NULL on allocation/input error.
 */
FloatTensor *st_from_sm(const FloatMatrix *src);

/**
 * @brief Convert float tensor to dense float matrix.
 * @param src Source float tensor (2D).
 * @return Dense float matrix, or NULL on allocation/input error.
 */
FloatMatrix *sm_from_st(const FloatTensor *src);

#endif  // ST_CONVERT_H
