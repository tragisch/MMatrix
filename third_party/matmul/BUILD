load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Optional: Konfigurierbare Includes & Libs für OpenBLAS
config_setting(
    name = "with_cblas",
    define_values = {"have_cblas": "true"},
)

cc_binary(
    name = "matmul",
    srcs = ["matmul.c"],
    copts = select({
        ":with_cblas": ["-DHAVE_CBLAS"],
        "//conditions:default": [],
    }),
    linkopts = select({
        ":with_cblas": ["-lopenblas"],
        "//conditions:default": [],
    }),
    deps = [],
    visibility = ["//visibility:public"],
)

perf_copts = [
    "-O3",
    "-fvectorize",
    "-ffast-math",
    "-funsafe-math-optimizations",
    "-funroll-loops",
    "-fomit-frame-pointer",
    "-march=native", # apple specific
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Wconversion",
    "-Wsign-conversion",
]

cc_library(
    name = "matmul_lib",
    srcs = ["matmul_lib.c"],
    hdrs = ["include/matmul_lib.h"],
    includes= ["include"],
    copts = perf_copts + select({
        ":with_cblas": ["-DHAVE_CBLAS"],
        "//conditions:default": [],
    }),
    linkopts = select({
        ":with_cblas": ["-lopenblas"],
        "//conditions:default": [],
    }),
    deps = [],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "matmul-boost",
    srcs = ["matmul-boost.cpp"],
    copts = ["-DNDEBUG", "-DBOOST_UBLAS_NDEBUG"],
    deps = [],
)

cc_binary(
    name = "matmul-eigen",
    srcs = ["matmul-eigen.cpp"],
    copts = ["-DNDEBUG"],
    deps = [],
)