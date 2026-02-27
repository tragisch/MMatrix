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

void dm_cplot(DoubleMatrix *mat);
void sm_cplot(FloatMatrix *mat);
void dms_cplot(DoubleSparseMatrix *mat, double strength);

/*******************************/
/*         Matlab Format       */
/*******************************/

// C API enums intentionally keep default underlying type for ABI stability.
// NOLINTBEGIN(performance-enum-size)

// MATLAB file format version.
typedef enum { MIO_FMT_MAT5, MIO_FMT_MAT73 } MIOFormat;

// MATLAB file compression mode.
typedef enum { MIO_COMPRESS_NONE, MIO_COMPRESS_ZLIB } MIOCompression;

// I/O specific status codes.
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

// Canonical status-string helper for I/O APIs.
const char *mio_status_to_string(MioStatus status);

// Global MATLAB I/O settings (defaults).
extern MIOFormat g_mio_format;
extern MIOCompression g_mio_compression;

// Set global MATLAB file format for subsequent writes (not thread-safe).
void mio_set_format(MIOFormat fmt);

// Get current global MATLAB file format.
MIOFormat mio_get_format(void);

// Set global MATLAB compression mode for writes (not thread-safe).
void mio_set_compression(MIOCompression comp);

// Get current global MATLAB compression mode.
MIOCompression mio_get_compression(void);

/*
 ====================================================================
 LEGACY I/O Functions (backward compatible, return int error code)
 ====================================================================
 */

// Legacy writer: returns 0 on success, non-zero on error.
int dm_write_mat_file(const DoubleMatrix *matrix, const char *filename);
MMATRIX_DEPRECATED("Use dm_write_mat_file instead")
int dm_write_MAT_file(const DoubleMatrix *matrix, const char *filename);

// Legacy writer: returns 0 on success, non-zero on error.
int sm_write_mat_file(const FloatMatrix *matrix, const char *filename);
MMATRIX_DEPRECATED("Use sm_write_mat_file instead")
int sm_write_MAT_file(const FloatMatrix *matrix, const char *filename);

// Legacy reader: returns matrix pointer or NULL on error.
DoubleMatrix *dm_read_mat_file(const char *filename);
MMATRIX_DEPRECATED("Use dm_read_mat_file instead")
DoubleMatrix *dm_read_MAT_file(const char *filename);

// Legacy reader: returns matrix pointer or NULL on error.
FloatMatrix *sm_read_mat_file(const char *filename);
MMATRIX_DEPRECATED("Use sm_read_mat_file instead")
FloatMatrix *sm_read_MAT_file(const char *filename);

// Legacy reader: returns matrix pointer or NULL on error.
DoubleSparseMatrix *dms_read_mat_file(const char *filename);
MMATRIX_DEPRECATED("Use dms_read_mat_file instead")
DoubleSparseMatrix *dms_read_MAT_file(const char *filename);

/*
 ====================================================================
 NEW I/O Functions (preferred, return MioStatus error codes)
 ====================================================================
 */

// Preferred writer: returns MioStatus and uses global format/compression settings.
MioStatus dm_write_mat_file_ex(const DoubleMatrix *matrix,
                               const char *filename);
MMATRIX_DEPRECATED("Use dm_write_mat_file_ex instead")
MioStatus dm_write_MAT_file_ex(const DoubleMatrix *matrix,
                               const char *filename);

// Preferred writer: returns MioStatus.
MioStatus sm_write_mat_file_ex(const FloatMatrix *matrix,
                               const char *filename);
MMATRIX_DEPRECATED("Use sm_write_mat_file_ex instead")
MioStatus sm_write_MAT_file_ex(const FloatMatrix *matrix,
                               const char *filename);

// Preferred reader: returns MioStatus and writes result to out_matrix.
MioStatus dm_read_mat_file_ex(const char *filename,
                              DoubleMatrix **out_matrix);
MMATRIX_DEPRECATED("Use dm_read_mat_file_ex instead")
MioStatus dm_read_MAT_file_ex(const char *filename,
                              DoubleMatrix **out_matrix);

// Preferred reader: returns MioStatus and writes result to out_matrix.
MioStatus sm_read_mat_file_ex(const char *filename,
                              FloatMatrix **out_matrix);
MMATRIX_DEPRECATED("Use sm_read_mat_file_ex instead")
MioStatus sm_read_MAT_file_ex(const char *filename,
                              FloatMatrix **out_matrix);

// Preferred reader: returns MioStatus and writes result to out_matrix.
MioStatus dms_read_mat_file_ex(const char *filename,
                               DoubleSparseMatrix **out_matrix);
MMATRIX_DEPRECATED("Use dms_read_mat_file_ex instead")
MioStatus dms_read_MAT_file_ex(const char *filename,
                               DoubleSparseMatrix **out_matrix);

DoubleSparseMatrix *dms_read_matrix_market(const char *filename);
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
