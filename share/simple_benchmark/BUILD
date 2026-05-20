load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_binary(
    name = "bench_dm",
    srcs = ["bench_dm.c"],
    deps = [
        ":simple_bench",
        "//app/matrix:dm",
    ],
)

cc_binary(
    name = "bench_sm",
    srcs = ["bench_sm.c"],
    deps = [
        ":simple_bench",
        "//app/matrix:sm",
    ],
)

cc_binary(
    name = "bench_dms",
    srcs = ["bench_dms.c"],
    deps = [
        ":simple_bench",
        "//app/matrix:dms",
    ],
)

cc_binary(
    name = "bench_dm_multiply_number",
    srcs = ["bench_dm_multiply_number.c"],
    deps = [
        ":simple_bench",
        "//app/matrix:dm",
    ],
)

cc_binary(
    name = "bench_dm_transpose",
    srcs = ["bench_dm_transpose.c"],
    deps = [
        ":simple_bench",
        "//app/matrix:dm",
    ],
)

cc_binary(
    name = "bench_st_conv",
    srcs = ["bench_st_conv.c"],
    deps = [
        ":simple_bench",
        "//app/tensor:st",
    ],
)

cc_binary(
    name = "bench_st_pool",
    srcs = ["bench_st_pool.c"],
    deps = [
        ":simple_bench",
        "//app/tensor:st",
    ],
)

cc_binary(
    name = "bench_sm_comparison",
    srcs = ["bench_sm_comparison.c"],
    deps = [
        ":simple_bench",
        "//app/matrix:sm",
    ],
)

cc_binary(
    name = "bench_sm_mps",
    srcs = ["bench_sm_mps.c"],
    linkopts = [
        "-framework Accelerate",
        "-lc++abi",
        "-framework Metal",
        "-framework MetalPerformanceShaders",
    ],
    target_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
    deps = [
        ":simple_bench",
        "//app/matrix:sm",
        "//app/matrix:sm_mps",
    ],
)

cc_binary(
    name = "bench_graphblas",
    srcs = ["bench_graphblas.c"],
    deps = [
        ":simple_bench",
        "@suitesparse",
    ],
)

cc_library(
    name = "simple_bench",
    srcs = [
        "simple_bench.h",
    ],
    copts = [
        "-O2",
        "-Wno-macro-redefined",
    ],
    visibility = ["//visibility:public"],
)
