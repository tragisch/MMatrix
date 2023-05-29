#ifndef DV_VECTOR_H
#define DV_VECTOR_H

#include "dm.h"
#include "dm_modify.h"

/*******************************/
/*  Double Vector  (Dynamic)   */
/*******************************/

// Create, Clone, Destroy
DoubleVector *dv_vector();
DoubleVector *dv_create(size_t length);
DoubleVector *dv_create_rand(size_t length);
DoubleVector *dv_create_from_array(const double *array, const size_t length);
DoubleVector *dv_clone(DoubleVector *vector);

// Getters and Setters
double *dv_get_array(const DoubleVector *vec);
void dv_set(DoubleVector *vec, size_t idx, double value);
double dv_get(const DoubleVector *vec, size_t idx);
void dv_push_value(DoubleVector *vec, double value);
double dv_pop_value(DoubleVector *vec);

// shrink, push, pop, expand
void dv_resize(DoubleVector *vec, size_t rows);

// free
void dv_destroy(DoubleVector *vec);

#endif // DV_VECTOR_H
