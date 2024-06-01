load("//:tools/matrix_build.bzl", "matrix_cc_binary")

matrix_cc_binary(
    name = "bench_dm_get",
    srcs = ["benchmark_dm_get.cpp"],
    deps = [
        "@com_google_benchmark//:benchmark",
    ],
)

matrix_cc_binary(
    name = "bench_dm_multiply",
    srcs = ["benchmark_dm_multiply_blas.cpp"],
    deps = [
        "@com_google_benchmark//:benchmark",
    ],
)