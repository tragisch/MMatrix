load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//tools:unity_test.bzl", "unity_test")
load("//tools/clang_doc:def.bzl", "clang_doc_target")

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
    name = "backend_accelerate_macos",
    constraint_values = [
        "@platforms//os:macos",
    ],
    values = {"define": "MATRIX_BACKEND=accelerate"},
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
    "//conditions:default": [],
})

package_copts = [
    "-DUSE_ACCELERATE",
    "-DACCELERATE_NEW_LAPACK",
]

openmp_copts = [
    "-Xpreprocessor",
    "-fopenmp",
]

openmp_linkopts = []

accelerate_linkopts = [
    "-framework Accelerate",
    "-lc++abi",
]

mps_linkopts = [
    "-framework Metal",
    "-framework MetalPerformanceShaders",
    "-framework MetalPerformanceShadersGraph",
]

default_deps = [
    "@openmp//:libomp",
    "@log",
]

clang_doc_target(
    name = "api_clang_doc",
    out = "api/st-clang-doc.tar",
    default_dest = "api/tensor",
    filter = ".*app/tensor/.*",
    format = "md",
    project_name = "MMatrix Tensor API",
    tags = [
        "manual",
        "no-sandbox",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "st",
    srcs = [
        "src/st.c",
        "src/st_backend.c",
        "src/st_backend.h",
        "src/st_batchnorm.c",
        "src/st_bf16_utils.h",
        "src/st_buffer.c",
        "src/st_buffer.h",
        "src/st_conv.c",
        "src/st_dtype.c",
        "src/st_dtype.h",
        "src/st_gpu_guard.c",
        "src/st_gpu_guard.h",
        "src/st_pool.c",
        "src/st_shape_ops.c",
        "src/st_warmup.c",
        "src/st_workspace.c",
        "src/st_workspace.h",
    ],
    hdrs = [
        "include/st.h",
        "include/st_batchnorm.h",
        "include/st_conv.h",
        "include/st_pool.h",
        "include/st_shape_ops.h",
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    local_defines = ["ST_INTERNAL_TENSOR_LAYOUT"],
    target_compatible_with = ["@platforms//os:macos"],
    visibility = ["//visibility:public"],
    deps = default_deps + [
        ":st_mps_objc",
        "//app/matrix:sm",
        "//app/matrix:sm_mps_objc",
    ],
)

cc_library(
    name = "st_internal_headers",
    hdrs = [
        "src/st_backend.h",
        "src/st_bf16_utils.h",
        "src/st_buffer.h",
        "src/st_dtype.h",
        "src/st_gpu_guard.h",
        "src/st_workspace.h",
    ],
    defines = ["ST_INTERNAL_TENSOR_LAYOUT"],
    includes = ["src"],
    visibility = ["//app/tensor:__pkg__"],
    deps = [
        ":st",
    ],
)

objc_library(
    name = "st_mps_objc",
    srcs = [
        "src/st_backend.h",
        "src/st_backend_mps.m",
        "src/st_buffer_metal.m",
        "src/st_mps.m",
        "src/st_stream_mps.m",
    ],
    hdrs = [
        "include/st.h",
        "include/st_conv.h",
        "include/st_pool.h",
        "src/st_buffer.h",
        "src/st_buffer_metal.h",
        "src/st_dtype.h",
        "src/st_mps.h",
        "src/st_stream_mps.h",
    ],
    copts = [
        "-ObjC",
        "-fobjc-arc",
    ],
    defines = ["ST_INTERNAL_TENSOR_LAYOUT"],
    includes = [
        "include",
        "src",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = ["//app/matrix:sm_mps_objc"],
)

unity_test(
    name = "test_st",
    srcs = ["tests/test_st.c"],
    features = ["coverage"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_conv",
    srcs = ["tests/test_st_conv.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_conv_backward",
    srcs = ["tests/test_st_conv_backward.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_pool",
    srcs = ["tests/test_st_pool.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_ops",
    srcs = ["tests/test_st_ops.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_batchnorm",
    srcs = ["tests/test_st_batchnorm.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_conv_backward_gemm",
    srcs = ["tests/test_st_conv_backward_gemm.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_batchnorm_relu",
    srcs = ["tests/test_st_batchnorm_relu.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_buffer",
    srcs = ["tests/test_st_buffer.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_workspace",
    srcs = ["tests/test_st_workspace.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_dtype",
    srcs = ["tests/test_st_dtype.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_gpu_guard",
    srcs = ["tests/test_st_gpu_guard.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_shape_ops",
    srcs = ["tests/test_st_shape_ops.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "example_cnn_dense_pipeline",
    srcs = ["tests/example_cnn_dense_pipeline.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_canonical",
    srcs = ["tests/bench_st_canonical.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_ab_conv",
    srcs = ["tests/bench_st_ab_conv.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_conv_medium_profile",
    srcs = ["tests/bench_st_conv_medium_profile.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_conv_medium_batch_profile",
    srcs = ["tests/bench_st_conv_medium_batch_profile.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_conv_backward_gemm",
    srcs = ["tests/bench_st_conv_backward_gemm.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_single_op",
    srcs = ["tests/bench_st_single_op.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_pipeline",
    srcs = ["tests/bench_st_pipeline.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_layout",
    srcs = ["tests/bench_st_layout.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_nhwc_vs_nchw",
    srcs = ["tests/bench_st_nhwc_vs_nchw.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_transfer",
    srcs = ["tests/bench_st_transfer.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_warmup",
    srcs = ["tests/bench_st_warmup.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_async_release",
    srcs = ["tests/bench_st_async_release.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_batchnorm",
    srcs = ["tests/bench_st_batchnorm.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

cc_binary(
    name = "bench_st_pool_diagnostic",
    srcs = ["tests/bench_st_pool_diagnostic.c"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)
