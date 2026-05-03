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
    name = "macos",
    constraint_values = [
        "@platforms//os:macos",
    ],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux",
    constraint_values = [
        "@platforms//os:linux",
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
    name = "backend_accelerate",
    values = {"define": "MATRIX_BACKEND=accelerate"},
)

config_setting(
    name = "backend_openblas",
    values = {"define": "MATRIX_BACKEND=openblas"},
)

config_setting(
    name = "backend_accelerate_macos",
    constraint_values = [
        "@platforms//os:macos",
    ],
    values = {"define": "MATRIX_BACKEND=accelerate"},
)

config_setting(
    name = "backend_openblas_linux",
    constraint_values = [
        "@platforms//os:linux",
    ],
    values = {"define": "MATRIX_BACKEND=openblas"},
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
    "-Wno-c23-extensions",
    "-Wno-unused-function",
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
    ":backend_accelerate_macos": [
        "-DUSE_ACCELERATE",
        "-DACCELERATE_NEW_LAPACK",
    ],
    ":backend_openblas_linux": [
        "-DUSE_OPENBLAS",
    ],
    "//conditions:default": [],
})

openmp_copts = select({
    ":macos": [
        "-Xpreprocessor",
        "-fopenmp",
    ],
    ":linux": ["-fopenmp"],
    "//conditions:default": [],
})

#############################
# Linker options (separat!)
#############################

openmp_linkopts = select({
    # On Linux we already link the OpenMP runtime through @openmp//:libomp.
    # Passing -fopenmp again at link time makes clang inject -lomp, which
    # bypasses Bazel's external dependency wiring and fails if libomp is not in
    # the system linker search path.
    ":linux": [],
    "//conditions:default": [],
})

accelerate_linkopts = select({
    ":backend_accelerate_macos": [
        "-framework Accelerate",
        "-lc++abi",
    ],
    "//conditions:default": [],
})

mps_linkopts = select({
    ":backend_accelerate_macos": [
        "-framework Metal",
        "-framework MetalPerformanceShaders",
        "-framework MetalPerformanceShadersGraph",
    ],
    "//conditions:default": [],
})

openblas_linkopts = []

#############################
# Dependencies
#############################

default_deps = [
    "@matio",
    "@openmp//:libomp",
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
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        ":backend_openblas": ["@platforms//os:linux"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + [
        ":dm",
        ":dms",
        ":sm",
        "//app/tensor:st",
    ],
)

cc_library(
    name = "dm",
    srcs = ["src/dm.c"],
    hdrs = ["include/dm.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        ":backend_openblas": ["@platforms//os:linux"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + select({
        ":backend_openblas_linux": ["@openblas"],
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
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        ":backend_openblas": ["@platforms//os:linux"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + select({
        ":backend_accelerate_macos": [":sm_mps_objc"],
        ":backend_openblas_linux": ["@openblas"],
        "//conditions:default": [],
    }),
)

objc_library(
    name = "sm_mps_objc",
    srcs = ["src/sm_mps.m"],
    hdrs = ["src/sm_mps.h"],
    copts = [
        "-ObjC",
        "-fobjc-arc",
    ],
    includes = ["src"],
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//app/tensor:__pkg__"],
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
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        ":backend_openblas": ["@platforms//os:linux"],
        "//conditions:default": [],
    }),
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
    data = ["tests/test_data/rand10.mat"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = ["@//app/matrix"],
)
