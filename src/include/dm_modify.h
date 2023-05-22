#ifndef DM_MODIFY_H
#define DM_MODIFY_H

#include "dm.h"

/*******************************/
/*           Modify            */
/*******************************/

// Get DoubleVector from DoubleMatrix:
DoubleVector *dm_get_row(DoubleMatrix *mat, size_t row_idx);
DoubleVector *dm_get_column(DoubleMatrix *mat, size_t column_idx);

void dm_set_column(DoubleMatrix *mat, size_t column_idx, DoubleVector *vec);
void dm_set_row(DoubleMatrix *mat, size_t row_idx, DoubleVector *vec);

// Push Vectors
void dm_push_column(DoubleMatrix *mat, DoubleVector *col_vec);
void dm_push_row(DoubleMatrix *mat, DoubleVector *row_vec);

// insert column vector
void dm_insert_column(DoubleMatrix *mat, size_t column_idx, DoubleVector *vec);
static void dm_insert_column_sparse(DoubleMatrix *mat, size_t column_idx);
static void dm_insert_column_dense(DoubleMatrix *mat, size_t column_idx);
static void dm_insert_column_hashtable(DoubleMatrix *mat, size_t column_idx);

DoubleMatrix *dm_get_sub_matrix(DoubleMatrix *mat, size_t row_start,
                                size_t row_end, size_t col_start,
                                size_t col_end);

/*******************************/
/*           Resize            */
/*******************************/

void dm_resize(DoubleMatrix *mat, size_t new_row, size_t new_col);

static void dm_resize_dense(DoubleMatrix *mat, size_t new_row, size_t new_col);
static void dm_resize_sparse(DoubleMatrix *mat, size_t new_row, size_t new_col);
static void dm_resize_hastable(DoubleMatrix *mat, size_t new_row,
                               size_t new_col);

#endif // DM_MODIFY_H