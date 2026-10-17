# `st_f16.h` – IEEE Binary16 Conversion API

Public API for IEEE binary16 (half precision) conversion.

## Functions

### `st_f32_to_f16`

`uint16_t st_f32_to_f16(float value)`

Convert one f32 value to IEEE binary16 bits.

Parameters:

- `value` (`float`): Input value.

Returns: binary16 bit pattern (round to nearest even, overflow to infinity, subnormals exact, NaN quiet with sign preserved).

### `st_f16_to_f32`

`float st_f16_to_f32(uint16_t bits)`

Convert IEEE binary16 bits to one f32 value.

Parameters:

- `bits` (`uint16_t`): binary16 bit pattern.

Returns: Exact f32 value (all binary16 values are exactly representable).

### `st_f32_to_f16_bulk`

`void st_f32_to_f16_bulk(const float *src, uint16_t *dst, size_t n)`

Convert a contiguous f32 buffer to binary16 bits.

Parameters:

- `src` (`const float *`): Source buffer with n values.
- `dst` (`uint16_t *`): Destination buffer with n uint16 slots.
- `n` (`size_t`): Number of elements.

### `st_f16_to_f32_bulk`

`void st_f16_to_f32_bulk(const uint16_t *src, float *dst, size_t n)`

Convert a contiguous binary16 buffer to f32.

Parameters:

- `src` (`const uint16_t *`): Source buffer with n uint16 slots.
- `dst` (`float *`): Destination buffer with n values.
- `n` (`size_t`): Number of elements.
