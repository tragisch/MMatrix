/*
 * Copyright (c) 2025 @tragisch <https://github.com/tragisch>
 * SPDX-License-Identifier: MIT
 *
 * This file is part of a project licensed under the MIT License.
 * See the LICENSE file in the root directory for details.
 */

#ifndef DM_IO_H
#define DM_IO_H

#include <matio.h>

#include "dm.h"
#include "dms.h"
#include "m_status.h"
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

/**
 * @enum MIOFormat
 * @brief MATLAB file format version.
 */
typedef enum { MIO_FMT_MAT5, MIO_FMT_MAT73 } MIOFormat;

/**
 * @enum MIOCompression
 * @brief Compression method for MATLAB files.
 */
typedef enum { MIO_COMPRESS_NONE, MIO_COMPRESS_ZLIB } MIOCompression;

// Globale Settings mit Default
static MIOFormat g_mio_format = MIO_FMT_MAT5;
static MIOCompression g_mio_compression = MIO_COMPRESS_NONE;

/**
 * @brief Set MATLAB file format for subsequent write operations.
 *
 * SEMANTICS:
 *   - Modifies global format setting (affects all subsequent dm_write_MAT_file calls).
 *   - MAT5: MATLAB 5 format (.mat files, widely compatible).
 *   - MAT73: MATLAB 7.3 format (HDF5-based, supports larger files).
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (modifies global state).
 *
 * @param fmt MIOFormat setting (MIO_FMT_MAT5 or MIO_FMT_MAT73).
 *
 * @see mio_get_format, mio_set_compression
 */
void mio_set_format(MIOFormat fmt);

/**
 * @brief Get current MATLAB file format setting.
 *
 * @return Current MIOFormat setting.
 *
 * @see mio_set_format
 */
MIOFormat mio_get_format(void);

/**
 * @brief Set compression method for MATLAB files.
 *
 * SEMANTICS:
 *   - Modifies global compression setting.
 *   - NONE: Faster write/read, larger file size.
 *   - ZLIB: Slower write/read, smaller file size.
 *
 * THREAD-SAFETY:
 *   NOT thread-safe (modifies global state).
 *
 * @param comp MIOCompression setting (MIO_COMPRESS_NONE or MIO_COMPRESS_ZLIB).
 *
 * @see mio_get_compression, mio_set_format
 */
void mio_set_compression(MIOCompression comp);

/**
 * @brief Get current compression setting.
 *
 * @return Current MIOCompression setting.
 *
 * @see mio_set_compression
 */
MIOCompression mio_get_compression(void);

/*
 ====================================================================
 LEGACY I/O Functions (backward compatible, return int error code)
 ====================================================================
 */

/**
 * @brief Write double matrix to MATLAB file (legacy, uses global format/compression).
 *
 * ERROR HANDLING:
 *   Returns 0 on success, non-zero on failure.
 *   Caller must interpret error code (not standardized).
 *
 * OWNERSHIP:
 *   File ownership: Function creates/overwrites file; caller owns result.
 *
 * DEPRECATED:
 *   Prefer dm_write_MAT_file_ex() which returns MStatus (structured error codes).
 *
 * @param matrix Matrix to write (const, not modified).
 * @param filename Output file path.
 *
 * @return 0 on success, non-zero on error (code varies by implementation).
 *
 * @see dm_write_MAT_file_ex, mio_set_format, mio_set_compression
 */
int dm_write_mat_file(const DoubleMatrix *matrix, const char *filename);
MMATRIX_DEPRECATED("Use dm_write_mat_file instead")
int dm_write_MAT_file(const DoubleMatrix *matrix, const char *filename);

/**
 * @brief Write float matrix to MATLAB file (legacy).
 *
 * @param matrix Matrix to write (const).
 * @param filename Output file path.
 *
 * @return 0 on success, non-zero on error.
 *
 * @see sm_write_MAT_file_ex
 */
int sm_write_mat_file(const FloatMatrix *matrix, const char *filename);
MMATRIX_DEPRECATED("Use sm_write_mat_file instead")
int sm_write_MAT_file(const FloatMatrix *matrix, const char *filename);

/**
 * @brief Read double matrix from MATLAB file (legacy, returns NULL on error).
 *
 * ERROR HANDLING:
 *   Returns NULL if file not found or read failed.
 *   No status information; caller cannot distinguish between error types.
 *
 * OWNERSHIP:
 *   Caller owns returned matrix; must call dm_destroy().
 *
 * DEPRECATED:
 *   Prefer dm_read_MAT_file_ex() which returns detailed MStatus.
 *
 * @param filename Input file path.
 *
 * @return Pointer to DoubleMatrix, or NULL if read failed.
 *
 * @see dm_read_MAT_file_ex, dm_destroy
 */
DoubleMatrix *dm_read_mat_file(const char *filename);
MMATRIX_DEPRECATED("Use dm_read_mat_file instead")
DoubleMatrix *dm_read_MAT_file(const char *filename);

/**
 * @brief Read float matrix from MATLAB file (legacy).
 *
 * @param filename Input file path.
 *
 * @return Pointer to FloatMatrix, or NULL if read failed.
 *
 * @see sm_read_MAT_file_ex
 */
FloatMatrix *sm_read_mat_file(const char *filename);
MMATRIX_DEPRECATED("Use sm_read_mat_file instead")
FloatMatrix *sm_read_MAT_file(const char *filename);

/**
 * @brief Read sparse double matrix from MATLAB file (legacy).
 *
 * @param filename Input file path.
 *
 * @return Pointer to DoubleSparseMatrix, or NULL if read failed.
 *
 * @see dms_read_MAT_file_ex
 */
DoubleSparseMatrix *dms_read_mat_file(const char *filename);
MMATRIX_DEPRECATED("Use dms_read_mat_file instead")
DoubleSparseMatrix *dms_read_MAT_file(const char *filename);

/*
 ====================================================================
 NEW I/O Functions (preferred, return MStatus error codes)
 ====================================================================
 */

/**
 * @brief Write double matrix to MATLAB file with detailed error reporting.
 *
 * SEMANTICS:
 *   - Uses global format/compression settings (set via mio_set_*).
 *   - Overwrites existing file if it exists.
 *
 * ERROR HANDLING:
 *   Returns MStatus enum (structured error codes):
 *   - MSTATUS_OK: Write successful
 *   - MSTATUS_IO_ERROR: File system error (permission, disk full)
 *   - MSTATUS_INVALID_ARGUMENT: Invalid parameters (NULL pointers, empty name)
 *
 * OWNERSHIP:
 *   File: Function creates/owns file; caller retains matrix ownership.
 *   Matrix: Not modified; caller still owns and must free.
 *
 * THREAD-SAFETY:
 *   Thread-safe (no global state modified); concurrent writes to different
 *   files safe. Concurrent writes to same file = undefined behavior (file lock needed).
 *
 * @param matrix Matrix to write (const, not modified).
 * @param filename Output file path (must not be NULL).
 *
 * @return MStatus code (MSTATUS_OK on success).
 *
 * @see mio_set_format, mio_set_compression, MStatus
 */
MStatus dm_write_mat_file_ex(const DoubleMatrix *matrix, const char *filename);
MMATRIX_DEPRECATED("Use dm_write_mat_file_ex instead")
MStatus dm_write_MAT_file_ex(const DoubleMatrix *matrix, const char *filename);

/**
 * @brief Write float matrix to MATLAB file with detailed error reporting.
 *
 * @param matrix Matrix to write (const).
 * @param filename Output file path.
 *
 * @return MStatus code.
 *
 * @see dm_write_MAT_file_ex
 */
MStatus sm_write_mat_file_ex(const FloatMatrix *matrix, const char *filename);
MMATRIX_DEPRECATED("Use sm_write_mat_file_ex instead")
MStatus sm_write_MAT_file_ex(const FloatMatrix *matrix, const char *filename);

/**
 * @brief Read double matrix from MATLAB file with detailed error reporting.
 *
 * SEMANTICS:
 *   - Allocates new DoubleMatrix and returns via output parameter.
 *   - If read fails, *out_matrix is set to NULL.
 *
 * ERROR HANDLING:
 *   Returns MStatus enum:
 *   - MSTATUS_OK: Read successful; *out_matrix = new matrix
 *   - MSTATUS_IO_ERROR: File not found or read failed
 *   - MSTATUS_ALLOC_FAILED: Memory allocation failed
 *
 * OWNERSHIP:
 *   - Caller owns returned matrix; must call dm_destroy() when done.
 *   - If status != MSTATUS_OK, *out_matrix = NULL (nothing to free).
 *
 * THREAD-SAFETY:
 *   Thread-safe (no global state modified).
 *   Multiple threads can read concurrently.
 *
 * @param filename Input file path (must not be NULL).
 * @param out_matrix Output parameter: receives pointer to new matrix.
 *                   Set to NULL if read failed.
 *
 * @return MStatus code (MSTATUS_OK on success).
 *
 * @see dm_destroy, MStatus
 */
MStatus dm_read_mat_file_ex(const char *filename, DoubleMatrix **out_matrix);
MMATRIX_DEPRECATED("Use dm_read_mat_file_ex instead")
MStatus dm_read_MAT_file_ex(const char *filename, DoubleMatrix **out_matrix);

/**
 * @brief Read float matrix from MATLAB file with detailed error reporting.
 *
 * @param filename Input file path.
 * @param out_matrix Output parameter (receives new matrix pointer).
 *
 * @return MStatus code.
 *
 * @see dm_read_MAT_file_ex
 */
MStatus sm_read_mat_file_ex(const char *filename, FloatMatrix **out_matrix);
MMATRIX_DEPRECATED("Use sm_read_mat_file_ex instead")
MStatus sm_read_MAT_file_ex(const char *filename, FloatMatrix **out_matrix);

/**
 * @brief Read sparse double matrix from MATLAB file with detailed error reporting.
 *
 * @param filename Input file path.
 * @param out_matrix Output parameter (receives new sparse matrix pointer).
 *
 * @return MStatus code.
 *
 * @see dm_read_MAT_file_ex
 */
MStatus dms_read_mat_file_ex(const char *filename,
                             DoubleSparseMatrix **out_matrix);
MMATRIX_DEPRECATED("Use dms_read_mat_file_ex instead")
MStatus dms_read_MAT_file_ex(const char *filename,
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
