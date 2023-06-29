#ifndef DM_MATH_BLAS_H
#define DM_MATH_BLAS_H

/*******************************/
/* Apples Accelarator or BLAS  */
/*******************************/

#ifdef __APPLE__
#include <Accelerate/Accelerate.h> // instead of cblas.h
#else
#include "cblas.h"
#endif

/*******************************/
/*   Matrix Multiplication     */
/*******************************/

#include "dm.h"
#include <stdlib.h>

#endif // DM_MATH_BLAS_H