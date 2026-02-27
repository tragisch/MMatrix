# API Return Conventions (Pragmatic Default)

This skill defines the default public error contract for MMatrix-style C APIs.

## Goal

Keep APIs easy for normal users while making failures explicit and testable.

## Default Return Rules

1. Allocating functions return pointer/NULL

- Return created object on success
- Return `NULL` on failure

Examples:

- `FloatMatrix *sm_add(const FloatMatrix *a, const FloatMatrix *b);`
- `DoubleMatrix *dm_multiply(const DoubleMatrix *a, const DoubleMatrix *b);`

2. Non-allocating functions that can fail return bool

- Return `true` on success
- Return `false` on failure

Examples:

- `bool sm_inplace_add(FloatMatrix *inout, const FloatMatrix *other);`
- `bool sm_gemm(FloatMatrix *C, float alpha, const FloatMatrix *A, SmTranspose ta, const FloatMatrix *B, SmTranspose tb, float beta);`

3. void only for infallible/debug operations

Examples:

- `void sm_print(const FloatMatrix *matrix);`

## Naming

- User-facing: `sm_<action>`, `dm_<action>`, `dms_<action>`
- In-place: `*_inplace_<action>`
- Advanced BLAS-style functions belong in an `Advanced Operations` section

## Optional Advanced Profile

For selected advanced modules, status enums (`NmStatus`, `SmStatus`, etc.) may be used where richer diagnostics are required.

## dm_/dms_ Migration Guidance

Apply the same contract to `dm_*` and `dms_*`:

- convert fallible `void` operations to `bool`
- keep allocating APIs as pointer/NULL
- keep debug helpers as `void`
- avoid silent failure paths
