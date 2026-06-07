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
    name = "st_runtime",
    srcs = [
        "src/runtime/st_buffer.c",
        "src/runtime/st_gpu_guard.c",
        "src/runtime/st_workspace.c",
    ],
    hdrs = [
        "include/st.h",
        "src/runtime/st_buffer.h",
        "src/runtime/st_gpu_guard.h",
        "src/runtime/st_workspace.h",
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = [
        "include",
        "src/backend",
        "src/backend/mps",
        "src/core",
        "src/runtime",
    ],
    local_defines = ["ST_INTERNAL_TENSOR_LAYOUT"],
    target_compatible_with = ["@platforms//os:macos"],
    visibility = ["//app/tensor:__pkg__"],
    deps = default_deps + [
        ":st_mps_objc",
    ],
)

cc_library(
    name = "st_core",
    srcs = [
        "src/core/st.c",
        "src/core/st_convert.c",
        "src/core/st_dtype.c",
        "src/core/st_shape_ops.c",
        "src/core/st_vector_bridge.c",
    ],
    hdrs = [
        "include/st.h",
        "include/st_convert.h",
        "include/st_shape_ops.h",
        "include/st_vector_bridge.h",
        "src/core/st_bf16_utils.h",
        "src/core/st_dtype.h",
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = [
        "include",
        "src/backend",
        "src/backend/mps",
        "src/core",
        "src/runtime",
    ],
    local_defines = ["ST_INTERNAL_TENSOR_LAYOUT"],
    target_compatible_with = ["@platforms//os:macos"],
    visibility = ["//app/tensor:__pkg__"],
    deps = default_deps + [
        "//app/matrix:sm",
        "//app/vector:sv",
        "//app/vector:vv",
        ":st_runtime",
    ],
)

cc_library(
    name = "st_backend",
    srcs = [
        "src/backend/mps/st_warmup.c",
        "src/backend/st_backend.c",
    ],
    hdrs = [
        "include/st.h",
        "src/backend/st_backend.h",
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = [
        "include",
        "src/backend",
        "src/backend/mps",
        "src/core",
        "src/runtime",
    ],
    local_defines = ["ST_INTERNAL_TENSOR_LAYOUT"],
    target_compatible_with = ["@platforms//os:macos"],
    visibility = ["//app/tensor:__pkg__"],
    deps = default_deps + [
        ":st_mps_objc",
    ],
)

cc_library(
    name = "st_ops_nn",
    srcs = [
        "src/ops/nn/st_batchnorm.c",
        "src/ops/nn/st_conv.c",
        "src/ops/nn/st_pool.c",
    ],
    hdrs = [
        "include/st_batchnorm.h",
        "include/st_conv.h",
        "include/st_pool.h",
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = [
        "include",
        "src/backend",
        "src/backend/mps",
        "src/core",
        "src/ops/nn",
        "src/runtime",
    ],
    local_defines = ["ST_INTERNAL_TENSOR_LAYOUT"],
    target_compatible_with = ["@platforms//os:macos"],
    visibility = ["//app/tensor:__pkg__"],
    deps = default_deps + [
        ":st_backend",
        ":st_core",
        ":st_runtime",
        "//app/matrix:sm",
    ],
)

cc_library(
    name = "st",
    hdrs = [
        "include/st.h",
        "include/st_batchnorm.h",
        "include/st_conv.h",
        "include/st_pool.h",
        "include/st_convert.h",
        "include/st_shape_ops.h",
        "include/st_vector_bridge.h",
    ],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = [
        "include",
        "src/backend",
        "src/backend/mps",
        "src/core",
        "src/ops/nn",
        "src/runtime",
    ],
    linkopts = openmp_linkopts + accelerate_linkopts + mps_linkopts,
    target_compatible_with = ["@platforms//os:macos"],
    visibility = ["//visibility:public"],
    deps = default_deps + [
        ":st_backend",
        ":st_core",
        ":st_ops_nn",
        ":st_runtime",
    ],
)

cc_library(
    name = "st_internal_headers",
    hdrs = [
        "src/backend/st_backend.h",
        "src/core/st_bf16_utils.h",
        "src/core/st_dtype.h",
        "src/runtime/st_buffer.h",
        "src/runtime/st_gpu_guard.h",
        "src/runtime/st_workspace.h",
    ],
    defines = ["ST_INTERNAL_TENSOR_LAYOUT"],
    includes = [
        "src/backend",
        "src/backend/mps",
        "src/core",
        "src/ops/nn",
        "src/runtime",
    ],
    visibility = ["//app/tensor:__pkg__"],
    deps = [
        ":st",
    ],
)

objc_library(
    name = "st_mps_objc",
    srcs = [
        "src/backend/mps/st_backend_mps.m",
        "src/backend/mps/st_mps.m",
        "src/backend/mps/st_stream_mps.m",
        "src/runtime/st_buffer_metal.m",
    ],
    hdrs = [
        "include/st.h",
        "include/st_conv.h",
        "include/st_pool.h",
        "src/backend/st_backend.h",
        "src/backend/mps/st_mps.h",
        "src/backend/mps/st_stream_mps.h",
        "src/core/st_dtype.h",
        "src/runtime/st_buffer.h",
        "src/runtime/st_buffer_metal.h",
    ],
    copts = [
        "-ObjC",
        "-fobjc-arc",
    ],
    defines = ["ST_INTERNAL_TENSOR_LAYOUT"],
    includes = [
        "include",
        "src/backend",
        "src/backend/mps",
        "src/core",
        "src/runtime",
    ],
    target_compatible_with = ["@platforms//os:macos"],
    deps = ["//app/matrix:sm_mps"],
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

unity_test(
    name = "test_st_convert",
    srcs = ["tests/test_st_convert.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":st",
        ":st_internal_headers",
    ],
)

unity_test(
    name = "test_st_vector_bridge",
    srcs = ["tests/test_st_vector_bridge.c"],
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
