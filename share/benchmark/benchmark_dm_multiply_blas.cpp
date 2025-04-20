
extern "C" {
#include "dm.h"
}

#include <benchmark/benchmark.h>

// A benchmark function
static void BM_dm_get(benchmark::State &state) {
  DoubleMatrix *dm1 = dm_create_random(50, 50, 0.001);
  dm_multiply_me_by_number(dm1, (double)arc4random_uniform(220));

  DoubleMatrix *dm2 = dm_create_random(50, 50, 0.001);
  dm_multiply_me_by_number(dm2, (double)arc4random_uniform(220));

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        dm_multiply(dm1, dm2)); // Benchmark the dm_get function
  }
  dm_destroy(dm1);
  dm_destroy(dm2);
}

// Register the benchmark function

BENCHMARK(BM_dm_get);

// Run the benchmark
BENCHMARK_MAIN();
