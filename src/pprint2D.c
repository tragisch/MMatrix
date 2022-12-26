/**
 * @file pprint2D.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "pprint2D.h"

#include <math.h>

#define MAX_ROW 20
#define MAX_ROW_PRINT 5
#define MAX_COLUMN 10
#define MAX_COLUMN_PRINT 4

/**
 * @brief printf DoubleArray pretty (instead of printVector)
 *
 * @param p_array
 * @param length
 */
void printDoubleArray_2(double* p_array, unsigned int length) {
  int zero = 0;

  if (length > 1) {
    for (size_t i = 0; i < length; i++) {
      if (fabs(p_array[i]) > 10e-8) zero = 1;
      if (i == 0) {
        if (zero) {
          printf("[%.2f ", p_array[i]);
        } else {
          printf("[.    ");
        }
      } else if (i == length - 1) {
        if (zero) {
          printf("%.2f]\n", p_array[i]);
        } else {
          printf(".   ]\n");
        }
      } else {
        if (length < MAX_COLUMN) {
          if (zero) {
            printf("%.2f ", p_array[i]);
          } else {
            printf(".    ");
          }
        } else {
          if (i < MAX_COLUMN_PRINT) {
            if (zero) {
              printf("%.2f ", p_array[i]);
            } else {
              printf(".    ");
            }
          } else if (i == MAX_COLUMN_PRINT) {
            printf(" ... ");
          } else if (i > length - MAX_COLUMN_PRINT - 1) {
            if (zero) {
              printf("%.2f ", p_array[i]);
            } else {
              printf(".    ");
            }
          }
        }
      }
    }
  } else {
    printf("[%.2f]\n", p_array[0]);
  }
}

int pprint_double(double num) {
  if (fabs(num) < 10e-16) {
    printf("0\t");
  } else {
    printf("%.2f", num);
  }

  return 0;
}

// Pretty print a 2D matrix in C (numpy style)
// https://codereview.stackexchange.com/questions/133999/pretty-print-a-2d-matrix-in-c-numpy-style
void pprint_numpy(fpoint_T* m, int row_size, int col_size) {
  int i, j;
  printf(" array([");

  if (row_size == 1 || col_size == 1) {
    int len = row_size == 1 ? col_size : row_size;

    if (col_size == 1) {
      for (i = 0; i < len; i++) {
        if (i == len - 1)
          printf("%.2e", m[i]);

        else {
          if (m[i] >= 0)
            printf(" %5.2e, ", m[i]);
          else
            printf("%5.2e, ", m[i]);
          if ((i + 1) % 6 == 0) printf("\n\t");
        }
      }
    } else {
      for (i = 0; i < len; i++) {
        if (i == 0)
          printf(" %.2e\n", m[i]);
        else if (i == len - 1)
          printf("\t %.2e", m[i]);

        else {
          if (m[i] >= 0)
            printf("\t %5.2e,\n", m[i]);
          else
            printf("\t %5.2e,\n", m[i]);
        }
      }
    }

    printf(" ])\n");

    return;
  }

  if (row_size > 10) {
    for (i = 0; i < 3; i++) {
      if (i == 0)
        printf("[ ");
      else
        printf("\t[ ");
      if (col_size > 10) {
        for (j = 0; j < 3; j++) {
          if (j < 2) {
            if (m[i * col_size + j] >= 0)
              printf(" %5.4e,\t", m[i * col_size + j]);
            else
              printf("%5.4e,\t", m[i * col_size + j]);
          } else {
            if (m[i * col_size + j] >= 0)
              printf(" %5.4e, ", m[i * col_size + j]);
            else
              printf("%5.4e, ", m[i * col_size + j]);
          }
        }

        printf("...,  ");

        if (m[i * col_size + col_size - 3] >= 0)
          printf(" %5.4e,\t\n\t", m[i * col_size + col_size - 3]);
        else
          printf("%5.4e,\t\n\t", m[i * col_size + col_size - 3]);

        if (m[i * col_size + col_size - 2] >= 0)
          printf("   %5.4e,\t", m[i * col_size + col_size - 2]);
        else
          printf("  %5.4e,\t", m[i * col_size + i]);

        if (m[i * col_size + col_size - 1] >= 0)
          printf(" %5.4e", m[i * col_size + col_size - 1]);
        else
          printf("%5.4e", m[i * col_size + col_size - 1]);
      }

      else {
        for (j = 0; j < col_size; j++) {
          if (j != col_size - 1)
            printf("%.6g, ", m[i * col_size + j]);
          else
            printf("%.6g", m[i * col_size + j]);
        }
      }
      printf(" ],\n");
    }

    printf("\t...,\n");

    for (i = row_size - 3; i < row_size; i++) {
      printf("\t[ ");
      if (col_size > 10) {
        for (j = 0; j < 3; j++) {
          if (j < 2) {
            if (m[i * col_size + j] >= 0)
              printf(" %5.4e,\t", m[i * col_size + j]);
            else
              printf("%5.4e,\t", m[i * col_size + j]);
          } else {
            if (m[i * col_size + j] >= 0)
              printf(" %5.4e, ", m[i * col_size + j]);
            else
              printf("%5.4e, ", m[i * col_size + j]);
          }
        }

        printf("...,  ");

        if (m[i * col_size + col_size - 3] >= 0)
          printf(" %5.4e,\t\n\t", m[i * col_size + col_size - 3]);
        else
          printf("%5.4e,\t\n\t", m[i * col_size + col_size - 3]);

        if (m[i * col_size + col_size - 2] >= 0)
          printf("   %5.4e,\t", m[i * col_size + col_size - 2]);
        else
          printf("  %5.4e,\t", m[i * col_size + i]);

        if (m[i * col_size + col_size - 1] >= 0)
          printf(" %5.4e", m[i * col_size + col_size - 1]);
        else
          printf("%5.4e", m[i * col_size + col_size - 1]);
      }

      else {
        for (j = 0; j < col_size; j++) {
          if (j != col_size - 1)
            printf("%.6g, ", m[i * col_size + j]);
          else
            printf("%.6g", m[i * col_size + j]);
        }
      }
      if (i == row_size - 1)
        printf(" ]])\n");
      else
        printf(" ],\n");
    }
  }

  else {
    for (i = 0; i < row_size; i++) {
      if (i == 0)
        printf("[ ");
      else
        printf("\t[ ");

      if (col_size > 10) {
        for (j = 0; j < 3; j++) {
          if (j < 2) {
            if (m[i * col_size + j] >= 0)
              printf(" %5.4e,\t", m[i * col_size + j]);
            else
              printf("%5.4e,\t", m[i * col_size + j]);
          } else {
            if (m[i * col_size + j] >= 0)
              printf(" %5.4e, ", m[i * col_size + j]);
            else
              printf("%5.4e, ", m[i * col_size + j]);
          }
        }

        printf("...,  ");

        if (m[i * col_size + col_size - 3] >= 0)
          printf(" %5.4e,\t\n", m[i * col_size + col_size - 3]);
        else
          printf("%5.4e,\t\n", m[i * col_size + col_size - 3]);

        if (m[i * col_size + col_size - 2] >= 0)
          printf("\t   %5.4e,\t", m[i * col_size + col_size - 2]);
        else
          printf("\t  %5.4e,\t", m[i * col_size + i]);

        if (m[i * col_size + col_size - 1] >= 0)
          printf(" %5.4e", m[i * col_size + col_size - 1]);
        else
          printf("%5.4e", m[i * col_size + col_size - 1]);
      }

      else {
        for (j = 0; j < col_size; j++) {
          if (j != col_size - 1)
            printf("%.6g, ", m[i * col_size + j]);
          else
            printf("%.6g", m[i * col_size + j]);
        }
      }
      if (i == row_size - 1)
        printf(" ]])\n");

      else
        printf(" ],\n");
    }
  }
}