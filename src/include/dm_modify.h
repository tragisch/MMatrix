#ifndef DM_MODIFY_H
#define DM_MODIFY_H

#include "dm.h"

/*******************************/
/*     Get / Set Row,Col       */
/*******************************/

// Get DoubleVector from DoubleMatrix:
DoubleVector *dm_get_row(DoubleMatrix *mat, size_t row_idx);
DoubleVector *dm_get_column(DoubleMatrix *mat, size_t column_idx);

// Set DoubleVector in DoubleMatrix:
void dm_set_column(DoubleMatrix *mat, size_t column_idx, DoubleVector *vec);
void dm_set_row(DoubleMatrix *mat, size_t row_idx, DoubleVector *vec);

/*******************************/
/*       Get SubMatrix         */
/*******************************/

// get sub matrix
DoubleMatrix *dm_get_sub_matrix(DoubleMatrix *mat, size_t row_start,
                                size_t row_end, size_t col_start,
                                size_t col_end);

/*******************************/
/*      Remove / Insert        */
/*******************************/

// remove entry from matrix (COO only)
void dm_remove_entry(DoubleMatrix *mat, size_t i, size_t j);
static void dm_remove_entry_coo(DoubleMatrix *mat, size_t i, size_t j);

// insert column vector
void dm_insert_column(DoubleMatrix *mat, size_t column_idx, DoubleVector *vec);
static void dm_insert_column_coo(DoubleMatrix *mat, size_t column_idx);
static void dm_insert_column_dense(DoubleMatrix *mat, size_t column_idx);

// insert row vector
void dm_insert_row(DoubleMatrix *mat, size_t row_idx, DoubleVector *vec);
static void dm_insert_row_coo(DoubleMatrix *mat, size_t row_idx);
static void dm_insert_row_dense(DoubleMatrix *mat, size_t row_idx);

// remove column vector
void dm_remove_column(DoubleMatrix *mat, size_t column_idx);
static void dm_remove_column_coo(DoubleMatrix *mat, size_t column_idx);
static void dm_remove_column_dense(DoubleMatrix *mat, size_t column_idx);

// remove row vector
void dm_remove_row(DoubleMatrix *mat, size_t row_idx);
static void dm_remove_row_coo(DoubleMatrix *mat, size_t row_idx);
static void dm_remove_row_dense(DoubleMatrix *mat, size_t row_idx);

/*******************************/
/*          Reshape            */
/*******************************/

// reshape matrix
void dm_reshape(DoubleMatrix *mat, size_t new_row, size_t new_col);
static void dm_reshape_coo(DoubleMatrix *mat, size_t new_row, size_t new_col);
static void dm_reshape_dense(DoubleMatrix *mat, size_t new_row, size_t new_col);

/*******************************/
/*           Resize            */
/*******************************/

void dm_resize(DoubleMatrix *mat, size_t new_row, size_t new_col);

static void dm_resize_dense(DoubleMatrix *mat, size_t new_row, size_t new_col);
static void dm_resize_coo(DoubleMatrix *mat, size_t new_row, size_t new_col);

/*******************************/
/*     Drop Small Entries      */
/*******************************/

void dm_drop_small_entries(DoubleMatrix *mat);
static void dm_drop_dense(DoubleMatrix *mat);
static void dm_drop_coo(DoubleMatrix *mat);

/*******************************/
/*          Order COO          */
/*******************************/

void dm_order_coo(DoubleMatrix *mat);
static void dm_swap_entries_coo(DoubleMatrix *mat, size_t i, size_t j);

#endif // DM_MODIFY_H