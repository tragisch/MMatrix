/**
 * @file st_vector_bridge.h
 * @brief Tensor-specific bridge helpers for vector views and copies.
 */

#ifndef ST_VECTOR_BRIDGE_H
#define ST_VECTOR_BRIDGE_H

#include "st.h"
#include "sv.h"
#include "vv.h"

FloatVectorView st_as_vv_view(FloatTensor *tensor);
FloatVector *st_to_sv(const FloatTensor *tensor);

#endif  // ST_VECTOR_BRIDGE_H
