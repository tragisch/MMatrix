# `st_convert.h` – Tensor/Matrix Conversion API

Conversion helpers between tensor and matrix types.

## Functions

### `st_from_sm`

`st_from_sm(const FloatMatrix *src)`

Convert dense float matrix to float tensor.

Parameters:

- `src` (`const FloatMatrix *`): Source float matrix.

Returns: Float tensor with shape [rows, cols], or NULL on allocation/input error.

### `sm_from_st`

`sm_from_st(const FloatTensor *src)`

Convert float tensor to dense float matrix.

Parameters:

- `src` (`const FloatTensor *`): Source float tensor (2D).

Returns: Dense float matrix, or NULL on allocation/input error.
