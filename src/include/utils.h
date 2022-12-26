/**
 * @file utils.h
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef UTILS_UR_H
#define UTILS_UR_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROW 20
#define MAX_ROW_PRINT 5
#define MAX_COLUMN 10
#define MAX_COLUMN_PRINT 4

/*******************************/
/*   Helper Functions          */
/*******************************/

// Functions Gereral:
// int randomInteger(int min, int max);
double randomNumber();

// Function DoubleArray
void printDoubleArray(double *p_array, unsigned int length);
double *createRandomDoubleArray(unsigned int length);

#endif  // !VECTOR_UR_H
