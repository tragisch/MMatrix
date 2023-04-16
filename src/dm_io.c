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
  FILE *file = fopen(filepath, "r");
  if (file == NULL) {
    return 1;
  }
  int succ_read = 1;
  for (size_t i = 0; i < vec->rows; i++) {
    // FIXME: insecure use of fscanf:
    succ_read = fscanf(file, "%15lf", &vec->values[i]);
  }
  fclose(file);

  return succ_read;
}

/* write data from DoubleVector to file */
int write_dm_vector_to_file(DoubleVector *vec, const char *filepath) {
  FILE *file = fopen(filepath, "w");
  if (file == NULL) {
    return 1;
  }

  for (size_t i = 0; i < vec->rows; i++) {
    if (i < vec->rows - 1) {
      fprintf(file, "%lf\n", vec->values[i]);
    } else {
      fprintf(file, "%lf", vec->values[i]);
    }
  }
  fclose(file);

  return 0;
}

/**
 * @brief printf DoubleArray pretty
 *
 * @param DoubleVector* vec
 */
void print_dm_vector(DoubleVector *vec) {
  double *array = dv_get_array(vec);
  size_t length = vec->rows;
  if (vec->rows == 1) {
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
    printf("Vector 1x%zi\n", vec->rows);
  } else {
    for (size_t i = 0; i < length; i++) {
      if ((i < MAX_COLUMN_PRINT) || (i > length - MAX_COLUMN_PRINT - 1)) {
        printf("[%.2lf]\n", array[i]);
      } else if (i == MAX_COLUMN_PRINT) {
        printf(" ... \n");
      }
    }
    printf("Vector 1x%zi\n", vec->rows);
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
  // print to console a DoubleMatrix matrix row by row with 2 digits precision
  for (size_t i = 0; i < matrix->rows; i++) {
    printf("[ ");
    for (size_t j = 0; j < matrix->cols; j++) {
      printf("%f ", matrix->values[i * matrix->cols + j]);
    }
    printf("]\n");
  }
  printf("Matrix %zix%zi\n", matrix->rows, matrix->cols);

  // if (matrix->rows < MAX_ROW) {
  //   for (size_t i = 0; i < matrix->rows; i++) {
  //     printDoubleArray(matrix->values[i], matrix->cols, 0);
  //   }
  // } else {
  //   for (size_t i = 0; i < 4; i++) {
  //     printDoubleArray(matrix->values[i], matrix->cols, 0);
  //   }
  //   printf("...\n");
  //   for (size_t i = matrix->rows - 4; i < matrix->rows; i++) {
  //     printDoubleArray(matrix->values[i], matrix->cols, 0);
  //   }
  // }

  // printf("Matrix %zix%zi, Capacity %zix%zi\n", matrix->rows, matrix->cols,
  //        matrix->rowCapacity, matrix->columnCapacity);
}
