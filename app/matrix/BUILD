load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//tools:unity_test.bzl", "unity_test")

config_setting(
    name = "coverage",
    values = {"collect_code_coverage": "true"},
)

#############################
# Define platforms
#############################

config_setting(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

#############################
# Test platform detection
#############################

genrule(
    name = "platform_check_genrule",
    outs = ["platform_check.txt"],
    cmd = select({
        ":macos_arm64": "echo 'Detected platform: macOS ARM64' > $@",
        ":linux_x86_64": "echo 'Detected platform: Linux x86_64' > $@",
        "//conditions:default": "echo 'Detected platform: Unknown or Default' > $@",
    }),
    visibility = ["//visibility:public"],
)

#############################
# Feature flags
#############################

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

#############################
# Compiler options
#############################

default_copts = [
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Wconversion",
    "-Wsign-conversion",
    "-Xpreprocessor",
]

perf_copts = select({
    ":macos_arm64": [
        "-O3",
        "-fvectorize",
        "-ffast-math",
        "-funsafe-math-optimizations",
        "-funroll-loops",
        "-fomit-frame-pointer",
        "-march=native",
    ],
    ":linux_x86_64": [
        "-O3",
        "-ffast-math",
        "-funsafe-math-optimizations",
        "-funroll-loops",
        "-fomit-frame-pointer",
        "-march=native",
        "-mtune=native",
        "-flto",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-falign-functions=32",
    ],
    "//conditions:default": [],
})

package_copts = select({
    ":use_accelerate": [
        "-DUSE_ACCELERATE",
        "-DACCELERATE_NEW_LAPACK",
    ],
    ":use_accelerate_mps": [
        "-DUSE_ACCELERATE_MPS",
        "-DACCELERATE_NEW_LAPACK",
        "-fobjc-arc",
    ],
    ":use_openblas": [
        "-DUSE_OPENBLAS",
    ],
    "//conditions:default": [],
})

openmp_copts = []  #"-fopenmp"

#############################
# Linker options (separat!)
#############################

openmp_linkopts = []

accelerate_linkopts = select({
    ":use_accelerate": [
        "-framework Accelerate",
        "-lc++abi",
    ],
    "//conditions:default": [],
})

mps_linkopts = select({
    ":use_accelerate_mps": [
        "-framework Accelerate",
        "-framework Metal",
        "-framework MetalPerformanceShaders",
        "-lc++abi",
    ],
    "//conditions:default": [],
})

openblas_linkopts = []

#############################
# Dependencies
#############################

default_deps = [
    "@matio",
    "@libomp",
    "@pcg",
    "@log",
]

#############################
# Header export
#############################

filegroup(
    name = "matrix_header",
    srcs = glob(["include/*.h"]),
    visibility = ["//visibility:public"],
)

#############################
# Simple Example
#############################

cc_binary(
    name = "matrix_example",
    srcs = ["matrix_example.c"],
    deps = [
        ":sm",
        "@openblas",
    ],
)

#############################
# Matrix libraries
#############################

cc_library(
    name = "matrix",
    srcs = [
        "src/m_convert.c",
        "src/m_io.c",
    ],
    hdrs = [
        "include/m_convert.h",
        "include/m_io.h",
        "include/m_status.h",
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    visibility = ["//visibility:public"],
    deps = default_deps + [
        ":dm",
        ":dms",
        ":sm",
    ],
)

cc_library(
    name = "dm",
    srcs = ["src/dm.c"],
    hdrs = ["include/dm.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    visibility = ["//visibility:public"],
    deps = default_deps + select({
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "sm",
    srcs = ["src/sm.c"],
    hdrs = ["include/sm.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    visibility = ["//visibility:public"],
    deps = default_deps + select({
        ":use_accelerate_mps": [":sm_mps_objc"],
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    }),
)

objc_library(
    name = "sm_mps_objc",
    srcs = ["src/sm_mps.m"],
    hdrs = ["include/sm_mps.h"],
    copts = ["-ObjC"],
    includes = ["."],
    target_compatible_with = select({
        ":use_accelerate_mps": ["@platforms//os:macos"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
)

cc_library(
    name = "dms",
    srcs = ["src/dms.c"],
    hdrs = ["include/dms.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts + [
        "-O3",
    ],
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    visibility = ["//visibility:public"],
    deps = default_deps + ["@suitesparse"],
)

#############################
# Unit tests
#############################

## Unit tests:

unity_test(
    name = "test_dm",
    srcs = ["tests/test_dm.c"],
    features = ["coverage"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = ["@//app/matrix:dm"],
)

unity_test(
    name = "test_sm",
    srcs = ["tests/test_sm.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = ["@//app/matrix:sm"],
)

unity_test(
    name = "test_dms",
    srcs = ["tests/test_dms.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = ["@//app/matrix:dms"],
)

unity_test(
    name = "test_convert",
    srcs = ["tests/test_convert.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = ["@//app/matrix"],
)

unity_test(
    name = "test_io",
    srcs = ["tests/test_io.c"],
    my_config = ["tests/my_config.yml"],
    data = ["tests/test_data/rand10.mat"],
    tags = ["smoke"],
    deps = ["@//app/matrix"],
)
