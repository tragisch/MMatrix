#ifndef DM_MATH_BLAS_H
#define DM_MATH_BLAS_H

/*******************************/
/* Apples Accelarator or BLAS  */
/*******************************/

#define APPLE_BLAS 1
#define BLAS 1

#ifdef BLAS
#ifndef APPLE_BLAS
#include "cblas.h"
#else
#include <Accelerate/Accelerate.h> // instead of cblas.h
#endif
#endif

/*******************************/
/*   Matrix Multiplication     */
/*******************************/

#include "dm.h"
#include <stdlib.h>

#endif // DM_MATH_BLAS_H