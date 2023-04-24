#ifndef DV_MATH_H
#define DV_MATH_H

#include <stdbool.h>

#include "dv_vector.h"
#include "misc.h"
#include "sp_math.h"

// #define NDEBUG

/*******************************/
/*     Define & Types         */
/*******************************/

// Definition of DoubleVector
typedef SparseMatrix DoubleVector;

/*******************************/
/*      Double Vector Math     */
/*******************************/

// Test equality:
bool dv_equal(DoubleVector *vec1, DoubleVector *vec2);

// math:
double dv_mean(const DoubleVector *vec);
double dv_min(const DoubleVector *vec);
double dv_max(const DoubleVector *vec);

void dv_add_vector(DoubleVector *vec1, const DoubleVector *vec2);
void dv_sub_vector(DoubleVector *vec1, const DoubleVector *vec2);
void dv_multiply_by_scalar(DoubleVector *vec, const double scalar);
void dv_divide_by_scalar(DoubleVector *vec, const double scalar);
void dv_add_constant(DoubleVector *vec, const double constant);
void dv_swap_elements(DoubleVector *vec, size_t idx_i, size_t idx_j);
void dv_reverse(DoubleVector *vec);
void dv_transpose(DoubleVector *vec);
double dv_dot_product(const DoubleVector *vec1, const DoubleVector *vec2);
double dv_magnitude(DoubleVector *vec);
void dv_normalize(DoubleVector *vec);

#endif // DV_MATH_H