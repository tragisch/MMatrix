/**
 * @file dm_print_pretty.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifdef __APPLE__
#include <stdlib.h>
#define random_number_generator arc4random
#else
#include <stdlib.h>
#include <time.h>
#define random_number_generator rand
#endif

#include "dm_math.h"
#include <float.h>

/*******************************/
/*         MIN/MAX DOUBLE          */
/*******************************/

double max_double(double a, double b) { return a > b ? a : b; }
double min_double(double a, double b) { return a < b ? a : b; }
int max_int(int a, int b) { return a > b ? a : b; }

bool is_zero(double value) { return fabs(value) < DBL_EPSILON; }

/*******************************/
/*       RANDOM DOUBLE         */
/*******************************/

double randomDouble() {
  uint32_t random_uint32 = randomInt();
  double random_double = (double)random_uint32 / (double)UINT32_MAX;
  return random_double;
}

double randomDouble_betweenBounds(uint32_t min, uint32_t max) {
  return (randomInt_betweenBounds(min, max - 1) + randomDouble());
}

/*******************************/
/*        RANDOM INT           */
/*******************************/

uint32_t randomInt() { return random_number_generator(); }

#ifdef __APPLE__

uint32_t randomInt_upperBound(uint32_t limit) {
  return arc4random_uniform(limit);
}

#else

uint32_t randomInt_upperBound(uint32_t limit) {
  static int initialized = 0;
  if (!initialized) {
    srand(time(NULL));
    initialized = 1;
  }
  int r;
  do {
    r = rand();
  } while (r >= RAND_MAX - RAND_MAX % limit);
  return r % limit;
}

#endif

uint32_t randomInt_betweenBounds(uint32_t min, uint32_t max) {
  if (max < min) {
    return min;
  }
  return randomInt_upperBound((max - min) + 1) + min;
}
