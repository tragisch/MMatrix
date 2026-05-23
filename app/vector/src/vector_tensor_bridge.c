/*
 * Copyright (c) 2026 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 */

#include "vector_tensor_bridge.h"

static FloatVectorView vv_invalid(void) { return vv_make(NULL, 0, 0); }

FloatVectorView st_as_vv_view(FloatTensor *tensor) {
  if (tensor == NULL || !st_is_contiguous(tensor) ||
      st_tensor_dtype(tensor) != ST_DTYPE_F32) {
    return vv_invalid();
  }
  return vv_make(st_tensor_mutable_data(tensor), st_tensor_numel(tensor), 1);
}

FloatVector *st_to_sv(const FloatTensor *tensor) {
  if (tensor == NULL || !st_is_contiguous(tensor) ||
      st_tensor_dtype(tensor) != ST_DTYPE_F32) {
    return NULL;
  }
  return sv_create_with_values(st_tensor_numel(tensor), st_tensor_data(tensor));
}
