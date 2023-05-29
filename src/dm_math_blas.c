/**
 * @file dm_math_blas.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 16-04-2023
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <assert.h>

#include "dm.h"
#include "dm_internals.h"
#include "dm_math.h"
#include "dm_math_blas.h"
#include "dv_vector.h"
#include <cblas.h>

/*******************************/
/*       CBLAS Support         */
/*******************************/

DoubleMatrix *dm_blas_multiply_by_matrix(const DoubleMatrix *mat1,
                                         const DoubleMatrix *mat2) {

  if (mat1->format != DENSE || mat2->format != DENSE) {
    perror("Error: BLAS only supports dense matrices only!");
    return NULL;
  }

  if (mat1->cols != mat2->rows) {
    perror(
        "Error: number of columns of m1 has to be euqal to number of rows of "
        "m2!");
    return NULL;
  }

  DoubleMatrix *product = dm_create(mat1->rows, mat2->cols);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (blasint)mat1->rows,
              (blasint)mat2->cols, (blasint)mat1->cols, 1.0, mat1->values,
              (blasint)mat1->cols, mat2->values, (blasint)mat2->cols, 0.0,
              product->values, (blasint)product->cols);

  return product;
}
