load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

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

# build options for different architectures
dm_copts = [
    "-Xpreprocessor",
    "-fopenmp",
] + perf_copts

sm_copts = [
    "-Xpreprocessor",
    "-fopenmp",
] + perf_copts

default_deps = [
    "@matio",
    "@libomp",
    "@pcg",
]

filegroup(
    name = "matrix_header",
    srcs = glob(["include/*.h"]),
    visibility = ["//visibility:public"],
)


cc_binary(
    name = "benchmark_sm_multiply",
    srcs = ["sm_benchmark_multiply.cpp"],
    deps = [
        ":sm",
        "@google_benchmark//:benchmark_main",
    ],
)

# matrix library
cc_library(
    name = "matrix",
    srcs = [
        "m_convert.c",
        "m_io.c",
    ],
    hdrs = [
        "include/m_convert.h",
        "include/m_io.h",
    ],
    copts = [
        "-O3",
        "-Wall", 
        "-Wextra",
        "-Wpedantic",
        "-Wconversion",
        "-Wsign-conversion"
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":dm",
        ":sm",
        ":dms",
    ],
)

# double matrix library
cc_library(
    name = "dm",
    srcs = [
        "dm.c",
    ],
    hdrs = [
        "include/dm.h",
    ],
    copts = dm_copts + select({
        ":use_accelerate": [
            "-DUSE_ACCELERATE",
            "-DACCELERATE_NEW_LAPACK",
        ],
        ":use_openblas": [
            "-DUSE_OPENBLAS",
        ],
        "//conditions:default": [],
    }),
    includes = ["include"],
    linkopts = select({
        ":use_accelerate": [
            "-framework Accelerate",
            "-lc++abi",
            "-lomp",
        ],
        "//conditions:default": ["-lomp"],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + select({
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    }),
)

# single matrix library (float)
cc_library(
    name = "sm",
    srcs = [
        "sm.c",
    ],
    hdrs = [
        "include/sm.h",
    ],
    copts = sm_copts + select({
        ":use_accelerate_mps": [
            "-DUSE_ACCELERATE_MPS",
            "-DACCELERATE_NEW_LAPACK",
            "-DUSE_MPS",
            "-fobjc-arc",
        ],
        ":use_openblas": [
            "-DUSE_OPENBLAS",
        ],
        ":use_accelerate": [
            "-DUSE_ACCELERATE",
            "-DACCELERATE_NEW_LAPACK",
        ],
        "//conditions:default": [],
    }),
    includes = ["include"],
    linkopts = select({
        ":use_accelerate_mps": [
            "-framework Accelerate",
            "-framework Metal",
            "-framework MetalPerformanceShaders",
            "-lc++abi",
            "-lomp",
        ],
        ":use_accelerate": [
            "-framework Accelerate",
            "-lc++abi",
            "-lomp",
        ],
        "//conditions:default": ["-lomp"],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + select({
        ":use_accelerate_mps": [":sm_mps_objc"],
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    }),
)

# Objective-C library for Metal Performance Shaders (on macOS)
objc_library(
    name = "sm_mps_objc",
    hdrs = ["sm_mps.h"],
    srcs = ["sm_mps.m"],
    copts = ["-ObjC"],
)


# sparse matrix (double) library
cc_library(
    name = "dms",
    srcs = [
        "dms.c",
    ],
    hdrs = [
        "include/dms.h",
    ],
    copts = [
    "-O3",
    "-Xpreprocessor",
    "-ffast-math",
    "-funsafe-math-optimizations",
    "-Wall", 
    "-Wextra",
    "-Wpedantic",
    "-Wconversion",
    "-Wsign-conversion"
    ],
    includes = ["include"],
    deps = default_deps + ["@suitesparse"],
    visibility = ["//visibility:public"],
)

# build configurations for different matrix libraries

config_setting(
    name = "use_accelerate",
    values = {"define": "USE_ACCELERATE=1"},
)

config_setting(
    name = "use_openblas",
    values = {"define": "USE_OPENBLAS=1"},
)

config_setting(
    name = "use_accelerate_mps",
    values = {"define": "USE_ACCELERATE_MPS=1"},
)
