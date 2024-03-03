
extern "C" {
#include "dm"
#include "dm.h"
#include "dm_math.h"
#include "dm_modify.h"
}

#include <benchmark/benchmark.h>

// A benchmark function
static void BM_dm_get(benchmark::State &state) {
  set_default_matrix_format(DENSE);
  DoubleMatrix *dm1 = dm_create_rand_between(50, 50, 0, 100, 0.001);
  DoubleMatrix *dm2 = dm_create_rand_between(50, 50, 0, 220, 0.001);

  for (auto _ : state) {
    benchmark::DoNotOptimize(dm_multiply_by_matrix(dm1,dm2))); // Benchmark the dm_get function
  }
  dm_destroy(dm);
}

// Register the benchmark function

BENCHMARK(BM_dm_get);

// Run the benchmark
BENCHMARK_MAIN();
