/**
 * @file sp_math.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 16-04-2023
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <assert.h>

#include "dbg.h"
#include "dm_matrix.h"
#include "sp_matrix.h"

/*******************************/
/*     Double Matrix Math      */
/*******************************/

/*
 * @brief get density of sparse matrix
 *
 * @param mat
 */

double sp_density(const SparseMatrix *mat) {
  return (double)mat->nnz / (mat->rows * mat->cols);
}
