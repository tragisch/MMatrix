load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//tools:unity_test.bzl", "unity_test")

config_setting(
    name = "coverage",
    values = {"collect_code_coverage": "true"},
)

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

config_setting(
    name = "backend_accelerate",
    values = {"define": "MATRIX_BACKEND=accelerate"},
)

config_setting(
    name = "backend_openblas",
    values = {"define": "MATRIX_BACKEND=openblas"},
)

default_copts = [
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Wconversion",
    "-Wsign-conversion",
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
    ":backend_accelerate": [
        "-DUSE_ACCELERATE",
        "-DACCELERATE_NEW_LAPACK",
    ],
    ":backend_openblas": [
        "-DUSE_OPENBLAS",
    ],
    "//conditions:default": [],
})

openmp_copts = select({
    ":macos_arm64": [
        "-Xpreprocessor",
        "-fopenmp",
    ],
    ":linux_x86_64": ["-fopenmp"],
    "//conditions:default": [],
})

openmp_linkopts = select({
    ":linux_x86_64": [],
    "//conditions:default": [],
})

accelerate_linkopts = select({
    ":backend_accelerate": [
        "-framework Accelerate",
        "-lc++abi",
    ],
    "//conditions:default": [],
})

mps_linkopts = select({
    ":backend_accelerate": [
        "-framework Metal",
        "-framework MetalPerformanceShaders",
        "-framework MetalPerformanceShadersGraph",
    ],
    "//conditions:default": [],
})

default_deps = [
    "@openmp//:libomp",
    "@log",
]

cc_library(
    name = "st",
    srcs = [
        "src/st.c",
        "src/st_batchnorm.c",
        "src/st_conv.c",
        "src/st_pool.c",
    ],
    hdrs = [
        "include/st.h",
        "include/st_batchnorm.h",
        "include/st_conv.h",
        "include/st_pool.h",
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    visibility = ["//visibility:public"],
    deps = default_deps + ["//app/matrix:sm"] + select({
        ":backend_accelerate": [
            ":st_mps_objc",
            "//app/matrix:sm_mps_objc",
        ],
        "//conditions:default": [],
    }),
)

objc_library(
    name = "st_mps_objc",
    srcs = ["src/st_mps.m"],
    hdrs = ["src/st_mps.h"],
    copts = [
        "-ObjC",
        "-fobjc-arc",
    ],
    includes = ["src"],
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = ["//app/matrix:sm_mps_objc"],
)

unity_test(
    name = "test_st",
    srcs = ["tests/test_st.c"],
    features = ["coverage"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

unity_test(
    name = "test_st_conv",
    srcs = ["tests/test_st_conv.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

unity_test(
    name = "test_st_conv_backward",
    srcs = ["tests/test_st_conv_backward.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

unity_test(
    name = "test_st_pool",
    srcs = ["tests/test_st_pool.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

unity_test(
    name = "test_st_ops",
    srcs = ["tests/test_st_ops.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

unity_test(
    name = "test_st_batchnorm",
    srcs = ["tests/test_st_batchnorm.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

unity_test(
    name = "test_st_conv_backward_gemm",
    srcs = ["tests/test_st_conv_backward_gemm.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

unity_test(
    name = "test_st_batchnorm_relu",
    srcs = ["tests/test_st_batchnorm_relu.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)
