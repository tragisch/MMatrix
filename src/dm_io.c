/**
 * @file dm_io.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm_io.h"

#include <assert.h>

#include "dbg.h"

// #define NDEBUG

/*******************************/
/*         I/O Functions       */
/*******************************/
#define MAX_ROW 20
#define MAX_ROW_PRINT 5
#define MAX_COLUMN 10
#define MAX_COLUMN_PRINT 4

/*******************************/
/*            Vector           */
/*******************************/

/* read in DoubleVector data from file */
int read_dm_vector_from_file(DoubleVector *vec, const char *filepath) {
  FILE *fp = fopen(filepath, "r");
  if (fp == NULL) {
    return 1;
  }
  int succ_read = 1;
  for (size_t i = 0; i < vec->length; i++) {
    // FIXME: insecure use of fscanf:
    succ_read = fscanf(fp, "%15lf", &vec->mat1D->values[i][0]);
  }
  fclose(fp);

  return succ_read;
}

/* write data from DoubleVector to file */
int write_dm_vector_to_file(DoubleVector *vec, const char *filepath) {
  FILE *fp = fopen(filepath, "w");
  if (fp == NULL) {
    return 1;
  }

  for (size_t i = 0; i < vec->length; i++) {
    if (i < vec->length - 1) {
      fprintf(fp, "%lf\n", vec->mat1D->values[i][0]);
    } else {
      fprintf(fp, "%lf", vec->mat1D->values[i][0]);
    }
  }
  fclose(fp);

  return 0;
}

/**
 * @brief printf DoubleArray pretty
 *
 * @param DoubleVector* vec
 */
void print_dm_vector(DoubleVector *vec) {
  double *array = get_array_from_vector(vec);
  size_t length = vec->mat1D->rows;
  if (vec->isColumnVector == false) {
    for (size_t i = 0; i < length; i++) {
      if (i == 0) {
        printf("[%.2lf ", array[i]);
      } else if (i == length - 1) {
        printf("%.2lf]\n", array[i]);
      } else {
        if (length < MAX_COLUMN) {
          printf("%.2lf ", array[i]);
        } else {
          if (i < MAX_COLUMN_PRINT) {
            printf("%.2lf ", array[i]);
          } else if (i == MAX_COLUMN_PRINT) {
            printf(" ... ");
          } else if (i > length - MAX_COLUMN_PRINT - 1) {
            printf("%.2lf ", array[i]);
          }
        }
      }
    }
    printf("Vector 1x%zi, Capacity %zi\n", vec->mat1D->rows,
           vec->mat1D->rowCapacity);
  } else {
    for (size_t i = 0; i < length; i++) {
      if ((i < MAX_COLUMN_PRINT) || (i > length - MAX_COLUMN_PRINT - 1)) {
        printf("[%.2lf]\n", array[i]);
      } else if (i == MAX_COLUMN_PRINT) {
        printf(" ... \n");
      }
    }
    printf("Vector 1x%zi, Capacity %zi\n", vec->mat1D->rows,
           vec->mat1D->rowCapacity);
  }
}

/*******************************/
/*            Matrix           */
/*******************************/

/**
 * @brief printf DoubleMatrix pretty
 *
 * @param num_cols
 * @param num_rows
 * @param matrix
 */
void print_dm_matrix(DoubleMatrix *matrix) {
  if (matrix->rows < MAX_ROW) {
    for (size_t i = 0; i < matrix->rows; i++) {
      printDoubleArray(matrix->values[i], matrix->columns, 0);
    }
  } else {
    for (size_t i = 0; i < 4; i++) {
      printDoubleArray(matrix->values[i], matrix->columns, 0);
    }
    printf("...\n");
    for (size_t i = matrix->rows - 4; i < matrix->rows; i++) {
      printDoubleArray(matrix->values[i], matrix->columns, 0);
    }
  }

  printf("Matrix %zix%zi, Capacity %zix%zi\n", matrix->rows, matrix->columns,
         matrix->rowCapacity, matrix->columnCapacity);
}
