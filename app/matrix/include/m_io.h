/**
 * @file m_io.h
 * @brief Public I/O and plotting API for matrix types (MAT + MatrixMarket).
 */

/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef DM_IO_H
#define DM_IO_H

#include "dm.h"
#include "dms.h"
#include "sm.h"

#ifndef MMATRIX_DEPRECATED
#if defined(__GNUC__) || defined(__clang__)
#define MMATRIX_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define MMATRIX_DEPRECATED(msg)
#endif
#endif

/*******************************/
/*     Plot DEFINES            */
/*******************************/

#define WIDTH (36)   // 44
#define HEIGHT (18)  // 22
#define X_DM (1)
#define Y_DM (1)
#define XMAX (WIDTH - X_DM - 1)
#define XMIN (1)  // -(WIDTH - X)
#define YMAX (HEIGHT - Y_DM - 1)
#define YMIN (1)  // -(HEIGHT - Y) + 1
#define MAX_NUMBER_OF_COLUMNS (30)
#define MAX_NUMBER_OF_ROWS (30)

/* ANSI escape codes for colors */
#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_COLOR_GREY_BASE "\x1b[48;5;%dm"

// extern char grid[HEIGHT][WIDTH];

/*******************************/
/*          I/O Functions      */
/*******************************/

/**
 * @brief Plot dense double matrix in terminal.
 * @param mat Matrix to display.
 */
void dm_cplot(DoubleMatrix *mat);
/**
 * @brief Plot dense float matrix in terminal.
 * @param mat Matrix to display.
 */
void sm_cplot(FloatMatrix *mat);
/**
 * @brief Plot sparse double matrix in terminal.
 * @param mat Sparse matrix to display.
 * @param strength Color intensity scale factor (typically 1.0).
 */
void dms_cplot(DoubleSparseMatrix *mat, double strength);

/*******************************/
/*         Matlab Format       */
/*******************************/

/** @brief C API enums keep default underlying type for ABI stability. */
// NOLINTBEGIN(performance-enum-size)

/** @brief MATLAB file format version. */
typedef enum { MIO_FMT_MAT5, MIO_FMT_MAT73 } MIOFormat;

/** @brief MATLAB file compression mode. */
typedef enum { MIO_COMPRESS_NONE, MIO_COMPRESS_ZLIB } MIOCompression;

/** @brief I/O-specific status codes. */
typedef enum MioStatus {
    MIO_STATUS_OK = 0,
    MIO_STATUS_INVALID_ARGUMENT = 1,
    MIO_STATUS_IO_ERROR = 2,
    MIO_STATUS_ALLOC_FAILED = 3,
    MIO_STATUS_FORMAT_ERROR = 4,
    MIO_STATUS_UNSUPPORTED_TYPE = 5,
    MIO_STATUS_INTERNAL_ERROR = 6,
} MioStatus;
// NOLINTEND(performance-enum-size)

/**
 * @brief Return canonical string for an I/O status code.
 * @param status Status code to convert.
 * @return Human-readable status string (static, caller must not free).
 */
const char *mio_status_to_string(MioStatus status);

/** @brief Global MATLAB I/O settings (defaults; not thread-safe). */
extern MIOFormat g_mio_format;
extern MIOCompression g_mio_compression;

/**
 * @brief Set global MATLAB file format for subsequent writes (not thread-safe).
 * @param fmt MATLAB file format (MIO_FMT_MAT5 or MIO_FMT_MAT73).
 */
void mio_set_format(MIOFormat fmt);

/**
 * @brief Get current global MATLAB file format.
 * @return Current MATLAB file format setting.
 */
MIOFormat mio_get_format(void);

/**
 * @brief Set global MATLAB compression mode for writes (not thread-safe).
 * @param comp Compression mode (MIO_COMPRESS_NONE or MIO_COMPRESS_ZLIB).
 */
void mio_set_compression(MIOCompression comp);

/**
 * @brief Get current global MATLAB compression mode.
 * @return Current compression mode setting.
 */
MIOCompression mio_get_compression(void);

/*
 ====================================================================
 LEGACY I/O Functions (backward compatible, return int error code)
 ====================================================================
 */

/**
 * @brief Legacy MAT writer for `DoubleMatrix` (`0` on success).
 * @param matrix Matrix to write.
 * @param filename Output file path.
 * @return 0 on success, non-zero error code on failure.
 */
int dm_write_mat_file(const DoubleMatrix *matrix, const char *filename);
/**
 * @deprecated Use dm_write_mat_file instead.
 * @param matrix Matrix to write.
 * @param filename Output file path.
 * @return 0 on success, non-zero error code on failure.
 */
MMATRIX_DEPRECATED("Use dm_write_mat_file instead")
int dm_write_MAT_file(const DoubleMatrix *matrix, const char *filename);

/**
 * @brief Legacy MAT writer for `FloatMatrix` (`0` on success).
 * @param matrix Matrix to write.
 * @param filename Output file path.
 * @return 0 on success, non-zero error code on failure.
 */
int sm_write_mat_file(const FloatMatrix *matrix, const char *filename);
/**
 * @deprecated Use sm_write_mat_file instead.
 * @param matrix Matrix to write.
 * @param filename Output file path.
 * @return 0 on success, non-zero error code on failure.
 */
MMATRIX_DEPRECATED("Use sm_write_mat_file instead")
int sm_write_MAT_file(const FloatMatrix *matrix, const char *filename);

/**
 * @brief Legacy MAT reader for `DoubleMatrix` (returns `NULL` on error).
 * @param filename Input MAT file path.
 * @return Pointer to read matrix, or NULL on error. Caller must free with dm_destroy.
 */
DoubleMatrix *dm_read_mat_file(const char *filename);
/**
 * @deprecated Use dm_read_mat_file instead.
 * @param filename Input MAT file path.
 * @return Pointer to read matrix, or NULL on error.
 */
MMATRIX_DEPRECATED("Use dm_read_mat_file instead")
DoubleMatrix *dm_read_MAT_file(const char *filename);

/**
 * @brief Legacy MAT reader for `FloatMatrix` (returns `NULL` on error).
 * @param filename Input MAT file path.
 * @return Pointer to read matrix, or NULL on error. Caller must free with sm_destroy.
 */
FloatMatrix *sm_read_mat_file(const char *filename);
/**
 * @deprecated Use sm_read_mat_file instead.
 * @param filename Input MAT file path.
 * @return Pointer to read matrix, or NULL on error.
 */
MMATRIX_DEPRECATED("Use sm_read_mat_file instead")
FloatMatrix *sm_read_MAT_file(const char *filename);

/**
 * @brief Legacy MAT reader for `DoubleSparseMatrix` (returns `NULL` on error).
 * @param filename Input MAT file path.
 * @return Pointer to read sparse matrix, or NULL on error. Caller must free with dms_destroy.
 */
DoubleSparseMatrix *dms_read_mat_file(const char *filename);
/**
 * @deprecated Use dms_read_mat_file instead.
 * @param filename Input MAT file path.
 * @return Pointer to read sparse matrix, or NULL on error.
 */
MMATRIX_DEPRECATED("Use dms_read_mat_file instead")
DoubleSparseMatrix *dms_read_MAT_file(const char *filename);

/*
 ====================================================================
 NEW I/O Functions (preferred, return MioStatus error codes)
 ====================================================================
 */

/**
 * @brief Preferred MAT writer for `DoubleMatrix` with global format/compression.
 * @param matrix Matrix to write.
 * @param filename Output file path.
 * @return Status code (MIO_STATUS_OK on success).
 */
MioStatus dm_write_mat_file_ex(const DoubleMatrix *matrix,
                               const char *filename);
/**
 * @deprecated Use dm_write_mat_file_ex instead.
 * @param matrix Matrix to write.
 * @param filename Output file path.
 * @return Status code.
 */
MMATRIX_DEPRECATED("Use dm_write_mat_file_ex instead")
MioStatus dm_write_MAT_file_ex(const DoubleMatrix *matrix,
                               const char *filename);

/**
 * @brief Preferred MAT writer for `FloatMatrix`.
 * @param matrix Matrix to write.
 * @param filename Output file path.
 * @return Status code (MIO_STATUS_OK on success).
 */
MioStatus sm_write_mat_file_ex(const FloatMatrix *matrix,
                               const char *filename);
/**
 * @deprecated Use sm_write_mat_file_ex instead.
 * @param matrix Matrix to write.
 * @param filename Output file path.
 * @return Status code.
 */
MMATRIX_DEPRECATED("Use sm_write_mat_file_ex instead")
MioStatus sm_write_MAT_file_ex(const FloatMatrix *matrix,
                               const char *filename);

/**
 * @brief Preferred MAT reader for `DoubleMatrix` (`out_matrix` receives result).
 * @param filename Input MAT file path.
 * @param out_matrix Output matrix pointer (allocated by function, caller must free).
 * @return Status code (MIO_STATUS_OK on success).
 */
MioStatus dm_read_mat_file_ex(const char *filename,
                              DoubleMatrix **out_matrix);
/**
 * @deprecated Use dm_read_mat_file_ex instead.
 * @param filename Input MAT file path.
 * @param out_matrix Output matrix pointer.
 * @return Status code.
 */
MMATRIX_DEPRECATED("Use dm_read_mat_file_ex instead")
MioStatus dm_read_MAT_file_ex(const char *filename,
                              DoubleMatrix **out_matrix);

/**
 * @brief Preferred MAT reader for `FloatMatrix` (`out_matrix` receives result).
 * @param filename Input MAT file path.
 * @param out_matrix Output matrix pointer (allocated by function, caller must free).
 * @return Status code (MIO_STATUS_OK on success).
 */
MioStatus sm_read_mat_file_ex(const char *filename,
                              FloatMatrix **out_matrix);
/**
 * @deprecated Use sm_read_mat_file_ex instead.
 * @param filename Input MAT file path.
 * @param out_matrix Output matrix pointer.
 * @return Status code.
 */
MMATRIX_DEPRECATED("Use sm_read_mat_file_ex instead")
MioStatus sm_read_MAT_file_ex(const char *filename,
                              FloatMatrix **out_matrix);

/**
 * @brief Preferred MAT reader for `DoubleSparseMatrix` (`out_matrix` receives result).
 * @param filename Input MAT file path.
 * @param out_matrix Output sparse matrix pointer (allocated by function, caller must free).
 * @return Status code (MIO_STATUS_OK on success).
 */
MioStatus dms_read_mat_file_ex(const char *filename,
                               DoubleSparseMatrix **out_matrix);
/**
 * @deprecated Use dms_read_mat_file_ex instead.
 * @param filename Input MAT file path.
 * @param out_matrix Output sparse matrix pointer.
 * @return Status code.
 */
MMATRIX_DEPRECATED("Use dms_read_mat_file_ex instead")
MioStatus dms_read_MAT_file_ex(const char *filename,
                               DoubleSparseMatrix **out_matrix);

/**
 * @brief Read sparse matrix from MatrixMarket file.
 * @param filename Input MatrixMarket file path (*.mtx).
 * @return Sparse matrix pointer (caller must free with dms_destroy), or NULL on error.
 */
DoubleSparseMatrix *dms_read_matrix_market(const char *filename);
/**
 * @brief Write sparse matrix to MatrixMarket file.
 * @param mat Sparse matrix to write.
 * @param filename Output MatrixMarket file path (*.mtx).
 */
void dms_write_matrix_market(const DoubleSparseMatrix *mat,
                             const char *filename);

/*******************************/
/*     Help Function Plot      */
/*******************************/

// static void init_grid(void);
// static void show_grid(DoubleMatrix *count);
// static int get_x_coord(size_t x, size_t rows);
// static int get_y_coord(size_t y, size_t cols);

#endif  // DM_IO_H
