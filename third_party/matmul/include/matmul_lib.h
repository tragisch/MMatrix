#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**********************************
 * Pseudo-random number generator *
 **********************************/

static uint64_t mat_rng[2] = {11ULL, 1181783497276652981ULL};

static inline uint64_t xorshift128plus(uint64_t s[2]);

double mat_drand(void);

/*******************************************
 * Helper routines for matrix manipulation *
 *******************************************/

float **mat_init(int n_rows, int n_cols);

void mat_destroy(float **m);

float **mat_gen_random(int n_rows, int n_cols);

float **mat_transpose(int n_rows, int n_cols, float *const *a);

float sdot_1(int n, const float *x, const float *y);

float sdot_8(int n, const float *x, const float *y);

#ifdef __SSE__
#include <xmmintrin.h>
float sdot_sse(int n, const float *x, const float *y);
#endif

/*************************
 * Matrix multiplication *
 *************************/

float **mat_mul0(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols,
                 float *const *b);

float **mat_mul1(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols,
                 float *const *b);

#ifdef __SSE__
float **mat_mul2(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols,
                 float *const *b);
float **mat_mul7(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols,
                 float *const *b);
#endif

float **mat_mul3(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols,
                 float *const *b);

float **mat_mul4(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols,
                 float *const *b);

#ifdef HAVE_CBLAS
#include <cblas.h>

float **mat_mul5(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols,
                 float *const *b);

float **mat_mul6(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols,
                 float *const *b);
#endif