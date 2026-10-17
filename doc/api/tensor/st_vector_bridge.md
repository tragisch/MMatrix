# `st_vector_bridge.h` – Tensor/Vector Bridge API

Tensor-specific bridge helpers for vector views and copies.

## Functions

### `st_as_vv_view`

`FloatVectorView st_as_vv_view(FloatTensor *tensor)`

Parameters:

- `tensor` (`FloatTensor *`)

### `st_to_sv`

`FloatVector * st_to_sv(const FloatTensor *tensor)`

Parameters:

- `tensor` (`const FloatTensor *`)
