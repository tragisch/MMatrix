load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

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
     #   "-DUSE_MPS",
        "-fobjc-arc",
    ],
    ":use_openblas": [
        "-DUSE_OPENBLAS",
    ],
    "//conditions:default": [],
})

openmp_copts = [] #"-fopenmp"

#############################
# Linker options (separat!)
#############################

openmp_linkopts = select({
    ":macos_arm64": ["-lomp"],
    ":linux_x86_64": ["-lgomp"],
    "//conditions:default": [],
})

accelerate_linkopts = select({
    ":use_accelerate": ["-framework Accelerate", "-lc++abi"],
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
    srcs = ["m_convert.c", "m_io.c"],
    hdrs = ["include/m_convert.h", "include/m_io.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":dm", ":sm", ":dms"],
)

cc_library(
    name = "dm",
    srcs = ["dm.c"],
    hdrs = ["include/dm.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = default_deps + select({
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "sm",
    srcs = ["sm.c"],
    hdrs = ["include/sm.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = default_deps + select({
        ":use_accelerate_mps": [":sm_mps_objc"],
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    }),
)

objc_library(
    name = "sm_mps_objc",
    hdrs = ["sm_mps.h"],
    srcs = ["sm_mps.m"],
    includes = ["."],
    copts = ["-ObjC"],
)

cc_library(
    name = "dms",
    srcs = ["dms.c"],
    hdrs = ["include/dms.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts + [
        "-O3",
    ],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts + openblas_linkopts,
    includes = ["include"],
    deps = default_deps + ["@suitesparse"],
    visibility = ["//visibility:public"],
)