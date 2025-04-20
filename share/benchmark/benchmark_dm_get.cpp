
extern "C" {
#include "dm.h"
#include "dm_modify.h"
}

#include <benchmark/benchmark.h>

// A benchmark function
static void BM_dm_get(benchmark::State &state) {
  set_default_matrix_format(DENSE);
  DoubleMatrix *dm_create_empty =
      dm_create_rand_between(500, 500, 0, 5,
                             0.0001); // Create the double matrix
  for (auto _ : state) {
    int i = rand() % dm_create_empty->rows;
    int j = rand() % dm_create_empty->cols;

    benchmark::DoNotOptimize(
        dm_get(dm_create_empty, i, j)); // Benchmark the dm_get function
  }
  dm_destroy(dm_create_empty);
}

// Register the benchmark function

BENCHMARK(BM_dm_get);

// Run the benchmark
BENCHMARK_MAIN();
