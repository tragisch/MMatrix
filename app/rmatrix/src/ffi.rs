#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use core::ffi::c_char;

#[repr(C)]
#[derive(Debug)]
pub struct FloatMatrix {
    pub rows: usize,
    pub cols: usize,
    pub capacity: usize,
    pub values: *mut f32,
}

extern "C" {
    pub fn sm_create_empty() -> *mut FloatMatrix;
    pub fn sm_create_zeros(rows: usize, cols: usize) -> *mut FloatMatrix;
    pub fn sm_create(rows: usize, cols: usize) -> *mut FloatMatrix;
    pub fn sm_create_with_values(rows: usize, cols: usize, values: *mut f32) -> *mut FloatMatrix;
    pub fn sm_clone(mat: *const FloatMatrix) -> *mut FloatMatrix;
    pub fn sm_create_identity(n: usize) -> *mut FloatMatrix;
    pub fn sm_create_random(rows: usize, cols: usize) -> *mut FloatMatrix;
    pub fn sm_create_random_he(rows: usize, cols: usize, fan_in: usize) -> *mut FloatMatrix;
    pub fn sm_create_random_xavier(
        rows: usize,
        cols: usize,
        fan_in: usize,
        fan_out: usize,
    ) -> *mut FloatMatrix;
    pub fn sm_from_array_ptrs(rows: usize, cols: usize, array: *mut *mut f32) -> *mut FloatMatrix;
    pub fn sm_from_array_static(rows: usize, cols: usize, array: *mut f32) -> *mut FloatMatrix;
    pub fn sm_create_array_from_matrix(matrix: *mut FloatMatrix) -> *mut f32;

    pub fn sm_get(mat: *const FloatMatrix, i: usize, j: usize) -> f32;
    pub fn sm_set(mat: *mut FloatMatrix, i: usize, j: usize, value: f32);
    pub fn sm_get_row(mat: *const FloatMatrix, i: usize) -> *mut FloatMatrix;
    pub fn sm_get_last_row(mat: *const FloatMatrix) -> *mut FloatMatrix;
    pub fn sm_get_col(mat: *const FloatMatrix, j: usize) -> *mut FloatMatrix;
    pub fn sm_get_last_col(mat: *const FloatMatrix) -> *mut FloatMatrix;
    pub fn sm_slice_rows(mat: *const FloatMatrix, start: usize, end: usize) -> *mut FloatMatrix;

    pub fn sm_reshape(matrix: *mut FloatMatrix, new_rows: usize, new_cols: usize);
    pub fn sm_resize(mat: *mut FloatMatrix, new_row: usize, new_col: usize);

    pub fn sm_transpose(mat: *const FloatMatrix) -> *mut FloatMatrix;

    pub fn sm_add(mat1: *const FloatMatrix, mat2: *const FloatMatrix) -> *mut FloatMatrix;
    pub fn sm_diff(mat1: *const FloatMatrix, mat2: *const FloatMatrix) -> *mut FloatMatrix;
    pub fn sm_multiply(mat1: *const FloatMatrix, mat2: *const FloatMatrix) -> *mut FloatMatrix;
    pub fn sm_multiply_4(A: *const FloatMatrix, B: *const FloatMatrix) -> *mut FloatMatrix;
    pub fn sm_elementwise_multiply(
        mat1: *const FloatMatrix,
        mat2: *const FloatMatrix,
    ) -> *mut FloatMatrix;
    pub fn sm_multiply_by_number(mat: *const FloatMatrix, number: f32) -> *mut FloatMatrix;
    pub fn sm_inverse(mat: *const FloatMatrix) -> *mut FloatMatrix;
    pub fn sm_div(mat1: *const FloatMatrix, mat2: *const FloatMatrix) -> *mut FloatMatrix;
    pub fn sm_solve_system(A: *const FloatMatrix, b: *const FloatMatrix) -> *mut FloatMatrix;

    pub fn sm_inplace_add(mat1: *mut FloatMatrix, mat2: *const FloatMatrix);
    pub fn sm_inplace_diff(mat1: *mut FloatMatrix, mat2: *const FloatMatrix);
    pub fn sm_inplace_square_transpose(mat: *mut FloatMatrix);
    pub fn sm_inplace_multiply_by_number(mat: *mut FloatMatrix, scalar: f32);
    pub fn sm_inplace_elementwise_multiply(mat1: *mut FloatMatrix, mat2: *const FloatMatrix);
    pub fn sm_inplace_div(mat1: *mut FloatMatrix, mat2: *const FloatMatrix);
    pub fn sm_inplace_normalize_rows(mat: *mut FloatMatrix);
    pub fn sm_inplace_normalize_cols(mat: *mut FloatMatrix);

    pub fn sm_determinant(mat: *const FloatMatrix) -> f32;
    pub fn sm_trace(mat: *const FloatMatrix) -> f32;
    pub fn sm_norm(mat: *const FloatMatrix) -> f32;
    pub fn sm_rank(mat: *const FloatMatrix) -> usize;
    pub fn sm_density(mat: *const FloatMatrix) -> f32;

    pub fn sm_is_empty(mat: *const FloatMatrix) -> bool;
    pub fn sm_is_square(mat: *const FloatMatrix) -> bool;
    pub fn sm_is_vector(mat: *const FloatMatrix) -> bool;
    pub fn sm_is_equal_size(mat1: *const FloatMatrix, mat2: *const FloatMatrix) -> bool;
    pub fn sm_is_equal(mat1: *const FloatMatrix, mat2: *const FloatMatrix) -> bool;
    pub fn sm_lu_decompose(mat: *mut FloatMatrix, pivot_order: *mut usize) -> bool;

    pub fn sm_print(matrix: *const FloatMatrix);
    pub fn sm_active_library() -> *const c_char;
    pub fn sm_destroy(mat: *mut FloatMatrix);
}
