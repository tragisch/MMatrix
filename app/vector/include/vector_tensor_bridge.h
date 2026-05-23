/**
 * @file vector_tensor_bridge.h
 * @brief Tensor-specific bridge helpers for vector views and copies.
 */

#ifndef VECTOR_TENSOR_BRIDGE_H
#define VECTOR_TENSOR_BRIDGE_H

#include "st.h"
#include "sv.h"
#include "vv.h"

FloatVectorView st_as_vv_view(FloatTensor *tensor);
FloatVector *st_to_sv(const FloatTensor *tensor);

#endif  // VECTOR_TENSOR_BRIDGE_H
