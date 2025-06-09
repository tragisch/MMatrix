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
#include "sm.h"

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

extern char grid[HEIGHT][WIDTH];

/*******************************/
/*          I/O Functions      */
/*******************************/

void dm_cplot(DoubleMatrix *mat);
void sm_cplot(FloatMatrix *mat);
void dms_cplot(DoubleSparseMatrix *mat, double strength);

/*******************************/
/*         Matlab Format       */
/*******************************/

typedef enum { MIO_FMT_MAT5, MIO_FMT_MAT73 } MIOFormat;

typedef enum { MIO_COMPRESS_NONE, MIO_COMPRESS_ZLIB } MIOCompression;

// Globale Settings mit Default
static MIOFormat g_mio_format = MIO_FMT_MAT5;
static MIOCompression g_mio_compression = MIO_COMPRESS_NONE;

void mio_set_format(MIOFormat fmt);
MIOFormat mio_get_format(void);

void mio_set_compression(MIOCompression comp);
MIOCompression mio_get_compression(void);

int dm_write_MAT_file(const DoubleMatrix *matrix, const char *filename);
int sm_write_MAT_file(const FloatMatrix *matrix, const char *filename);
DoubleMatrix *dm_read_MAT_file(const char *filename, const char *varname);
FloatMatrix *sm_read_MAT_file(const char *filename, const char *varname);

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
