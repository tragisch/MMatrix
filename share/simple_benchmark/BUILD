load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

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
