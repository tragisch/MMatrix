/**
 * @file utils.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.2
 * @date 26-12-2ß22
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "utils.h"

/*******************************/
/*   Helper Functions          */
/*******************************/

/**
 * @brief return random number between 0 ... 1
 *
 * @return double
 */
double randomNumber() { return (double)arc4random() / (double)RAND_MAX; }

/*******************************/
/*        Double Array       */
/*******************************/

/**
 * @brief Create a DoubleArray object in HEAP
 *
 * @param length
 * @param value
 * @return double*
 */
double* createRandomDoubleArray(unsigned int length) {
  double* array = (double*)calloc(length, sizeof(double));

  for (unsigned int i = 0; i < length; i++) {
    array[i] = randomNumber();
  }
  return array;  // die Speicheradresse wird zurückgegeben.
}

/**
 * @brief printf DoubleArray pretty (instead of printVector)
 *
 * @param p_array
 * @param length
 */
void printDoubleArray(double* p_array, unsigned int length) {
  if (length > 1) {
    for (size_t i = 0; i < length; i++) {
      if (i == 0) {
        printf("[%.2f ", p_array[i]);
      } else if (i == length - 1) {
        printf("%.2f]\n", p_array[i]);
      } else {
        if (length < MAX_COLUMN) {
          printf("%.2f ", p_array[i]);
        } else {
          if (i < MAX_COLUMN_PRINT) {
            printf("%.2f ", p_array[i]);
          } else if (i == MAX_COLUMN_PRINT) {
            printf(" ... ");
          } else if (i > length - MAX_COLUMN_PRINT - 1) {
            printf("%.2f ", p_array[i]);
          }
        }
      }
    }
  } else {
    printf("[%lf]\n", p_array[0]);
  }
}