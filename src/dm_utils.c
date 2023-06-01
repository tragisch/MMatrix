/**
 * @file dm_utils.c
 * @author Uwe Röttgermann (uwe@roettgermann.de)
 * @brief
 * @version 0.1
 * @date 26-12-2022
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "dm.h"
#include "dm_io.h"
#include "dm_modify.h"
#include <stdio.h>
#include <time.h>

/*******************************/
/*  performance measurement    */
/*******************************/

// write double date to file in a table woth row "name" and colum format
void write_double_to_file(char *filename, char *name, double data,
                          matrix_format format, double density) {
  FILE *fp = NULL;
  fp = fopen(filename, "a");
  if (fp == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }
  time_t tm = 0;
  time(&tm);
  fprintf(fp, "%s\t%.10lf\t%lf\t%d\t%s\n", name, data, density, format,
          ctime(&tm));
  fclose(fp);
}

/*******************************/
/*         Progress Bar        */
/*******************************/

// if file to read is very large, print a progress bar:
void print_progress_bar(size_t progress, size_t total, int barWidth) {
  float percentage = (float)progress / (float)total;
  int filledWidth = (int)(percentage * (float)barWidth);

  printf("[");
  for (int i = 0; i < barWidth; i++) {
    if (i < filledWidth) {
      printf("=");
    } else {
      printf(" ");
    }
  }
  printf("] %d%%\r", (int)(percentage * 100));
  fflush(stdout);
}

/*******************************/
/*     max index array       */
/*******************************/

size_t find_max_index(const size_t *data, size_t length) {
  size_t max_index = 0;
  size_t max_value = data[0];

  for (size_t i = 1; i < length; i++) {
    if (data[i] > max_value) {
      max_value = data[i];
      max_index = i;
    }
  }

  return max_index;
}
