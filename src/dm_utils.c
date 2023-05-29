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

/*******************************/
/*  performance measurement    */
/*******************************/

// write double date to file in a table woth row "name" and colum format
void write_double_to_file(char *filename, char *name, double data,
                          matrix_format format) {
  FILE *fp;
  fp = fopen(filename, "a");
  if (fp == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }
  fprintf(fp, "%s\t%.10lf\t%d\n", name, data, format);
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
