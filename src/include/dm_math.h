#ifndef DM_MATH_H
#define DM_MATH_H

#include "dm.h"
#include <math.h>

/*******************************/
/*      General stuff     */
/*******************************/

double max_double(double a, double b);
double min_double(double a, double b);
int max_int(int a, int b);
bool is_zero(double value);

/*******************************/
/*  Random Functions           */
/*******************************/

double randomDouble();
double randomDouble_betweenBounds(uint32_t min, uint32_t max);
uint32_t randomInt();
uint32_t randomInt_betweenBounds(uint32_t min, uint32_t max);
uint32_t randomInt_upperBound(uint32_t limit);

/*******************************/
/* Matrix*Matrix Multiplication*/
/*******************************/

DoubleMatrix *dm_multiply_by_matrix(const DoubleMatrix *mat1,
                                    const DoubleMatrix *mat2);

// use cblas_dgemm to multiply two matrices
static DoubleMatrix *dm_multiply_by_matrix_blas(const DoubleMatrix *mat1,
                                                const DoubleMatrix *mat2);

static DoubleMatrix *dm_multiply_by_matrix_coo(const DoubleMatrix *matrixA,
                                               const DoubleMatrix *matrixB);

// general but naive apprach (not used, too slow)
static DoubleMatrix *dm_multiply_by_matrix_naive(const DoubleMatrix *mat1,
                                                 const DoubleMatrix *mat2);

// helper for coo
static void accumulate_result(DoubleMatrix *result, size_t row, size_t col,
                              double value);

/*******************************/
/* Matrix*Vector Multiplication */
/*******************************/

DoubleVector *dm_multiply_by_vector(const DoubleMatrix *mat,
                                    const DoubleVector *vec);
static DoubleVector *dm_multiply_by_vector_naive(const DoubleMatrix *mat,
                                                 const DoubleVector *vec);
static DoubleVector *dm_multiply_by_vector_blas(const DoubleMatrix *mat,
                                                const DoubleVector *vec);
static DoubleVector *dm_multiply_by_vector_coo(const DoubleMatrix *mat,
                                               const DoubleVector *vec);
static DoubleVector *dm_multiply_by_vector_csc(const DoubleMatrix *mat,
                                               const DoubleVector *vec);

/*******************************/
/* Matrix*Scalar Multiplication */
/*******************************/

void dm_multiply_by_scalar(DoubleMatrix *mat, const double scalar);
static void dm_multiply_by_scalar_sparse(DoubleMatrix *mat,
                                         const double scalar);
static void dm_multiply_by_scalar_dense(DoubleMatrix *mat, const double scalar);
static void dm_multiply_by_scalar_coo(DoubleMatrix *mat, const double scalar);
static void dm_multiply_by_scalar_csc(DoubleMatrix *mat, const double scalar);

/*******************************/
/* Matrix*Scalar Multiplication */
/*******************************/

bool dm_equal(const DoubleMatrix *m1, const DoubleMatrix *m2);
void dm_transpose(DoubleMatrix *mat);
double dm_determinant(const DoubleMatrix *mat);
double dm_density(const DoubleMatrix *mat);
double dm_trace(const DoubleMatrix *mat);

// principles
DoubleMatrix *dm_inverse(DoubleMatrix *mat);
size_t dm_rank(const DoubleMatrix *mat);

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

#endif // DM_MATH_H
