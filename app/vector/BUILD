load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tools:unity_test.bzl", "unity_test")

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

openblas_linkopts = []

default_deps = [
    "@log",
    "@openmp//:libomp",
]

filegroup(
    name = "vector_header",
    srcs = glob(["include/*.h"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sv",
    srcs = ["src/sv.c"],
    hdrs = ["include/sv.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + openblas_linkopts,
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
    name = "dv",
    srcs = ["src/dv.c"],
    hdrs = ["include/dv.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + openblas_linkopts,
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
    name = "vv",
    srcs = ["src/vv.c"],
    hdrs = ["include/vv.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + openblas_linkopts,
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        ":backend_openblas": ["@platforms//os:linux"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + [
        ":sv",
    ] + select({
        ":backend_openblas_linux": ["@openblas"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "dvs",
    srcs = ["src/dvs.c"],
    hdrs = ["include/dvs.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + openblas_linkopts,
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        ":backend_openblas": ["@platforms//os:linux"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + [
        ":dv",
    ] + select({
        ":backend_openblas_linux": ["@openblas"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "vec3",
    srcs = ["src/vec3.c"],
    hdrs = ["include/vec3.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + openblas_linkopts,
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        ":backend_openblas": ["@platforms//os:linux"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps,
)

cc_library(
    name = "vector_bridge",
    srcs = ["src/vector_bridge.c"],
    hdrs = ["include/vector_bridge.h"],
    copts = default_copts + perf_copts + package_copts + openmp_copts,
    includes = ["include"],
    linkopts = openmp_linkopts + accelerate_linkopts + openblas_linkopts,
    target_compatible_with = select({
        ":backend_accelerate": ["@platforms//os:macos"],
        ":backend_openblas": ["@platforms//os:linux"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + [
        ":dv",
        ":dvs",
        ":sv",
        ":vv",
        "//app/matrix:dm",
        "//app/matrix:dms",
        "//app/matrix:sm",
    ] + select({
        ":backend_openblas_linux": ["@openblas"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "vector",
    hdrs = glob(["include/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":dv",
        ":dvs",
        ":sv",
        ":vec3",
        ":vector_bridge",
        ":vv",
    ],
)

unity_test(
    name = "test_sv",
    srcs = ["tests/test_sv.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":sv"],
)

unity_test(
    name = "test_dv",
    srcs = ["tests/test_dv.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":dv"],
)

unity_test(
    name = "test_vv",
    srcs = ["tests/test_vv.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":sv",
        ":vector_bridge",
        ":vv",
    ],
)

unity_test(
    name = "test_dvs",
    srcs = ["tests/test_dvs.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":dv",
        ":dvs",
    ],
)

unity_test(
    name = "test_vec3",
    srcs = ["tests/test_vec3.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [":vec3"],
)

unity_test(
    name = "test_vector_bridge",
    srcs = ["tests/test_vector_bridge.c"],
    my_config = ["tests/my_config.yml"],
    tags = ["smoke"],
    deps = [
        ":sv",
        ":vector_bridge",
    ],
)

cc_binary(
    name = "bench_sv_dv",
    srcs = ["tests/bench_sv_dv.c"],
    copts = default_copts + perf_copts + package_copts,
    linkopts = accelerate_linkopts + openblas_linkopts,
    target_compatible_with = ["@platforms//os:macos"],
    deps = [
        ":dv",
        ":sv",
    ],
)
