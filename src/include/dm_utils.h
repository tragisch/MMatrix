#ifndef DM_UTILS_H
#define DM_UTILS_H

#include "dm.h"
#include <stdio.h>
#include <time.h>

// from stackoverflow:
// https://stackoverflow.com/questions/5248915/execution-time-of-c-program:
#define CPUTIME(FCALL)                                                         \
  ({                                                                           \
    double START = clock();                                                    \
    FCALL;                                                                     \
    ((double)clock() - START) / CLOCKS_PER_SEC;                                \
  })

void write_double_to_file(char *filename, char *name, double data,
                          matrix_format format, double density);

void print_progress_bar(size_t progress, size_t total, int barWidth);

size_t find_max_index(const size_t *data, size_t length);

#endif // DM_UTILS_H