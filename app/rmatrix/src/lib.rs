pub mod ffi;

use core::ffi::{c_char, c_void};
use std::{
    ffi::CStr,
    fmt,
    ptr::NonNull,
    str::Utf8Error,
};

pub type Result<T> = std::result::Result<T, MatrixError>;

#[derive(Debug)]
pub enum MatrixError {
    NullPointer(&'static str),
    LengthMismatch { expected: usize, actual: usize },
    DimensionOverflow { rows: usize, cols: usize },
    Utf8(Utf8Error),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::NullPointer(ctx) => write!(f, "{} returned a null pointer", ctx),
            MatrixError::LengthMismatch { expected, actual } => write!(
                f,
                "length mismatch: expected {} elements but received {}",
                expected, actual
            ),
            MatrixError::DimensionOverflow { rows, cols } => write!(
                f,
                "matrix dimensions {}x{} overflow when computing total length",
                rows, cols
            ),
            MatrixError::Utf8(err) => write!(f, "utf8 conversion error: {}", err),
        }
    }
}

impl std::error::Error for MatrixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MatrixError::Utf8(err) => Some(err),
            _ => None,
        }
    }
}

impl From<Utf8Error> for MatrixError {
    fn from(value: Utf8Error) -> Self {
        Self::Utf8(value)
    }
}

extern "C" {
    fn free(ptr: *mut c_void);
}

#[derive(Debug)]
pub struct Matrix {
    raw: NonNull<ffi::FloatMatrix>,
}

impl Matrix {
    /// Construct a Matrix from a raw pointer returned by the C API.
    /// The pointer must own the allocation.
    unsafe fn from_raw(ptr: *mut ffi::FloatMatrix, context: &'static str) -> Result<Self> {
        NonNull::new(ptr)
            .map(|raw| Self { raw })
            .ok_or(MatrixError::NullPointer(context))
    }

    pub fn zeros(rows: usize, cols: usize) -> Result<Self> {
        unsafe { Self::from_raw(ffi::sm_create_zeros(rows, cols), "sm_create_zeros") }
    }

    pub fn identity(size: usize) -> Result<Self> {
        unsafe { Self::from_raw(ffi::sm_create_identity(size), "sm_create_identity") }
    }

    pub fn random(rows: usize, cols: usize) -> Result<Self> {
        unsafe { Self::from_raw(ffi::sm_create_random(rows, cols), "sm_create_random") }
    }

    pub fn from_slice(rows: usize, cols: usize, values: &[f32]) -> Result<Self> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(MatrixError::DimensionOverflow { rows, cols })?;

        if values.len() != expected {
            return Err(MatrixError::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }

        unsafe {
            Self::from_raw(
                ffi::sm_create_with_values(rows, cols, values.as_ptr() as *mut f32),
                "sm_create_with_values",
            )
        }
    }

    pub fn try_clone(&self) -> Result<Self> {
        unsafe { Self::from_raw(ffi::sm_clone(self.as_ptr()), "sm_clone") }
    }

    pub fn rows(&self) -> usize {
        unsafe { self.raw.as_ref().rows }
    }

    pub fn cols(&self) -> usize {
        unsafe { self.raw.as_ref().cols }
    }

    pub fn capacity(&self) -> usize {
        unsafe { self.raw.as_ref().capacity }
    }

    pub fn len(&self) -> usize {
        self.rows() * self.cols()
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        unsafe { ffi::sm_get(self.as_ptr(), row, col) }
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        unsafe { ffi::sm_set(self.as_mut_ptr(), row, col, value) }
    }

    pub fn transpose(&self) -> Result<Self> {
        unsafe { Self::from_raw(ffi::sm_transpose(self.as_ptr()), "sm_transpose") }
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        unsafe { Self::from_raw(ffi::sm_add(self.as_ptr(), rhs.as_ptr()), "sm_add") }
    }

    pub fn diff(&self, rhs: &Self) -> Result<Self> {
        unsafe { Self::from_raw(ffi::sm_diff(self.as_ptr(), rhs.as_ptr()), "sm_diff") }
    }

    pub fn multiply(&self, rhs: &Self) -> Result<Self> {
        unsafe { Self::from_raw(ffi::sm_multiply(self.as_ptr(), rhs.as_ptr()), "sm_multiply") }
    }

    pub fn multiply_elementwise(&self, rhs: &Self) -> Result<Self> {
        unsafe {
            Self::from_raw(
                ffi::sm_elementwise_multiply(self.as_ptr(), rhs.as_ptr()),
                "sm_elementwise_multiply",
            )
        }
    }

    pub fn multiply_scalar(&self, scalar: f32) -> Result<Self> {
        unsafe {
            Self::from_raw(
                ffi::sm_multiply_by_number(self.as_ptr(), scalar),
                "sm_multiply_by_number",
            )
        }
    }

    pub fn determinant(&self) -> f32 {
        unsafe { ffi::sm_determinant(self.as_ptr()) }
    }

    pub fn trace(&self) -> f32 {
        unsafe { ffi::sm_trace(self.as_ptr()) }
    }

    pub fn norm(&self) -> f32 {
        unsafe { ffi::sm_norm(self.as_ptr()) }
    }

    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let count = self.len();
        unsafe {
            let data = ffi::sm_create_array_from_matrix(self.as_ptr() as *mut _);
            let data = NonNull::new(data).ok_or(MatrixError::NullPointer(
                "sm_create_array_from_matrix",
            ))?;
            let slice = std::slice::from_raw_parts(data.as_ptr(), count);
            let mut vec = Vec::with_capacity(count);
            vec.extend_from_slice(slice);
            free(data.as_ptr().cast::<c_void>());
            Ok(vec)
        }
    }

    pub fn as_ptr(&self) -> *const ffi::FloatMatrix {
        self.raw.as_ptr()
    }

    pub fn as_mut_ptr(&self) -> *mut ffi::FloatMatrix {
        self.raw.as_ptr()
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe { ffi::sm_destroy(self.raw.as_ptr()) }
    }
}

pub fn active_library() -> Result<String> {
    unsafe {
        let ptr = ffi::sm_active_library();
        let ptr = NonNull::new(ptr as *mut c_char)
            .ok_or(MatrixError::NullPointer("sm_active_library"))?;
        let c_str = CStr::from_ptr(ptr.as_ptr());
        Ok(c_str.to_str()?.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_multiplication() -> Result<()> {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let matrix = Matrix::from_slice(2, 2, &data)?;
        let identity = Matrix::identity(2)?;
        let result = matrix.multiply(&identity)?;
        assert_eq!(result.to_vec()?, data);
        Ok(())
    }

    #[test]
    fn transpose_changes_shape() -> Result<()> {
        let matrix = Matrix::from_slice(2, 3, &[1., 2., 3., 4., 5., 6.])?;
        let transposed = matrix.transpose()?;
        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.cols(), 2);
        Ok(())
    }
}
