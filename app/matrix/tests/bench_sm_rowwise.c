/*
 * Row-helper benchmarks against scalar nm_lib-style loops.
 * Run: bazel run //app/matrix:bench_sm_rowwise
 */

#include "sm.h"

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef enum RowOp {
  ROW_ADD,
  ROW_SUM,
  ROW_ARGMAX,
} RowOp;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static double median(double *samples, size_t count) {
  for (size_t i = 1; i < count; ++i) {
    const double value = samples[i];
    size_t j = i;
    while (j > 0 && samples[j - 1] > value) {
      samples[j] = samples[j - 1];
      --j;
    }
    samples[j] = value;
  }
  return samples[count / 2];
}

static size_t repetitions(size_t count) {
  if (count <= 8192) return 128;
  if (count <= 65536) return 32;
  return 4;
}

static FloatMatrix *scalar_sum_rows(const FloatMatrix *mat) {
  FloatMatrix *out = sm_create(1, mat->cols);
  if (!out) return NULL;
  for (size_t col = 0; col < mat->cols; ++col) {
    float sum = 0.0f;
    for (size_t row = 0; row < mat->rows; ++row) {
      sum += mat->values[row * mat->cols + col];
    }
    out->values[col] = sum;
  }
  return out;
}

static FloatMatrix *scalar_argmax_rows(const FloatMatrix *mat) {
  FloatMatrix *out = sm_create(mat->rows, 1);
  if (!out) return NULL;
  for (size_t row = 0; row < mat->rows; ++row) {
    size_t best = 0;
    for (size_t col = 1; col < mat->cols; ++col) {
      if (mat->values[row * mat->cols + col] >
          mat->values[row * mat->cols + best]) {
        best = col;
      }
    }
    out->values[row] = (float)best;
  }
  return out;
}

static const char *operation_name(RowOp op) {
  switch (op) {
    case ROW_ADD: return "add_rowwise";
    case ROW_SUM: return "sum_rows";
    case ROW_ARGMAX: return "argmax_rows";
  }
  return "unknown";
}

static double measure(bool optimized, RowOp op, FloatMatrix *mat,
                      const FloatMatrix *row, size_t reps) {
  double samples[7];
  for (size_t sample = 0; sample < 7; ++sample) {
    const uint64_t start = now_ns();
    for (size_t r = 0; r < reps; ++r) {
      if (op == ROW_ADD) {
        if (optimized) {
          if (!sm_add_rowwise(mat, row)) return -1.0;
        } else {
          for (size_t i = 0; i < mat->rows; ++i) {
            for (size_t j = 0; j < mat->cols; ++j) {
              mat->values[i * mat->cols + j] += row->values[j];
            }
          }
        }
      } else {
        FloatMatrix *out = NULL;
        if (optimized) {
          out = op == ROW_SUM ? sm_sum_rows(mat) : sm_argmax_rows(mat);
        } else {
          out = op == ROW_SUM ? scalar_sum_rows(mat) : scalar_argmax_rows(mat);
        }
        if (!out) return -1.0;
        sm_destroy(out);
      }
    }
    samples[sample] = (double)(now_ns() - start) / 1000.0 / (double)reps;
  }
  return median(samples, 7);
}

int main(void) {
  const size_t row_counts[] = {32, 256, 1024, 4096};
  const size_t cols = 256;
  const int max_threads = omp_get_max_threads();
  const int thread_counts[] = {1, max_threads};
  omp_set_dynamic(0);

  printf("operation,rows,cols,threads,scalar_us,sm_us,speedup\n");
  for (size_t si = 0; si < sizeof(row_counts) / sizeof(row_counts[0]); ++si) {
    const size_t rows = row_counts[si];
    const size_t count = rows * cols;
    FloatMatrix *mat = sm_create(rows, cols);
    FloatMatrix *bias = sm_create(1, cols);
    if (!mat || !bias) {
      fprintf(stderr, "allocation failed for %zux%zu\n", rows, cols);
      sm_destroy(mat);
      sm_destroy(bias);
      return 1;
    }
    for (size_t i = 0; i < count; ++i) {
      mat->values[i] = (float)((i * 31u) % 997u) * 0.001f;
    }
    for (size_t i = 0; i < cols; ++i) {
      bias->values[i] = (float)i * 0.0001f;
    }

    for (size_t ti = 0; ti < 2; ++ti) {
      const int threads = thread_counts[ti];
      if (ti == 1 && threads == thread_counts[0]) continue;
      omp_set_num_threads(threads);
      for (int op = ROW_ADD; op <= ROW_ARGMAX; ++op) {
        const size_t reps = repetitions(count);
        const double before =
            measure(false, (RowOp)op, mat, bias, reps);
        const double after =
            measure(true, (RowOp)op, mat, bias, reps);
        printf("%s,%zu,%zu,%d,%.3f,%.3f,%.2fx\n",
               operation_name((RowOp)op), rows, cols, threads, before, after,
               after > 0.0 ? before / after : 0.0);
      }
    }
    sm_destroy(bias);
    sm_destroy(mat);
  }
  omp_set_num_threads(max_threads);
  return 0;
}