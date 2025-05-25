load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")

platform(
    name = "macos_arm64_platform",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "on_macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

# build options for different architectures
dm_copts = [
    "-O3",
    "-Xpreprocessor", # needs to accept -fopenmp
    "-fopenmp",
    "-Wall", 
    "-Wextra",
    "-Wpedantic",
    "-Wconversion",
    "-Wsign-conversion"
]

sm_copts = [
    "-O3",
    "-ffast-math",
    "-funsafe-math-optimizations",
    "-Xpreprocessor", # needs to accept -fopenmp
    "-fopenmp",
    "-Wall", 
    "-Wextra", 
    "-Wpedantic",
    "-Wconversion",
    "-Wsign-conversion"
]

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
        ":on_macos_arm64": [
            "-DUSE_ACCELERATE",
            "-DACCELERATE_NEW_LAPACK",
        ],
        "//conditions:default": [],
    }) + select({
        ":use_openblas": [
            "-DUSE_OPENBLAS",
        ],
        "//conditions:default": [],
    }),


    includes = ["include"],

    linkopts = select({
        ":on_macos_arm64": [
            "-framework Accelerate",
            "-lc++abi",
            "-lomp",
        ],
        "//conditions:default": ["-lomp"],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": default_deps + [],
        ":use_accelerate": default_deps + [],
        ":use_accelerate_mps": default_deps +[],
        ":use_openblas": default_deps + ["@openblas"],
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
        ":on_macos_arm64": [
            "-DUSE_ACCELERATE_MPS",
            "-DACCELERATE_NEW_LAPACK",
            "-DUSE_MPS",
            "-fobjc-arc",
        ],
        "//conditions:default": [],
    }) + select({
        ":use_openblas": [
            "-DUSE_OPENBLAS",
        ],
        "//conditions:default": [],
    }),
    includes = ["include"],
    linkopts = select({
        ":on_macos_arm64": [
            "-framework Accelerate",
            "-framework Metal",
            "-framework MetalPerformanceShaders",
            "-lc++abi",
            "-lomp",
        ],
        "//conditions:default": ["-lomp"],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + [":sm_mps_objc"] + select({
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    })
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




