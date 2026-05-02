load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
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

default_copts = [
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Wconversion",
    "-Wsign-conversion",
    "-Wno-c23-extensions",
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

openmp_linkopts = select({
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

default_deps = [
    "@openmp//:libomp",
    "@log",
]

cc_library(
    name = "st",
    srcs = [
        "src/st.c",
        "src/st_backend.c",
        "src/st_batchnorm.c",
        "src/st_buffer.c",
        "src/st_conv.c",
        "src/st_dtype.c",
        "src/st_pool.c",
        "src/st_workspace.c",
    ],
    hdrs = [
        "include/st.h",
        "include/st_backend.h",
        "include/st_batchnorm.h",
        "include/st_buffer.h",
        "include/st_conv.h",
        "include/st_dtype.h",
        "include/st_pool.h",
        "include/st_workspace.h",
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        ":backend_openblas": ["@platforms//os:linux"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + ["//app/matrix:sm"] + select({
        ":backend_accelerate_macos": [
            ":st_mps_objc",
            "//app/matrix:sm_mps_objc",
        ],
        "//conditions:default": [],
    }),
)

objc_library(
    name = "st_mps_objc",
    srcs = [
        "src/st_backend_mps.m",
        "src/st_buffer_metal.m",
        "src/st_mps.m",
    ],
    hdrs = [
        "include/st.h",
        "include/st_backend.h",
        "include/st_buffer.h",
        "include/st_conv.h",
        "src/st_buffer_metal.h",
        "src/st_mps.h",
    ],
    copts = [
        "-ObjC",
        "-fobjc-arc",
    ],
    includes = [
        "include",
        "src",
    ],
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

unity_test(
    name = "test_st_buffer",
    srcs = ["tests/test_st_buffer.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

unity_test(
    name = "test_st_workspace",
    srcs = ["tests/test_st_workspace.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

unity_test(
    name = "test_st_dtype",
    srcs = ["tests/test_st_dtype.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":st"],
)

cc_binary(
    name = "bench_st_conv_backward_gemm",
    srcs = ["tests/bench_st_conv_backward_gemm.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [":st"],
)
