# load("@rules_cc//cc:cc_library.bzl", "cc_library")

# # homebrew installed local openblas
# cc_library(
#     name = "openblas",
#     srcs = ["lib/libopenblas.a"],
#     hdrs = glob([
#         "include/*.h",
#     ]),
#     includes = [
#         "include"
#     ],
#     visibility = ["//visibility:public"],
# )

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

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

filegroup(
    name = "all-src",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "openblas",
    generate_args = select({
        ":macos_arm64": [
            "-DDYNAMIC_ARCH=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DUSE_OPENMP=ON",
            "-DUSE_THREAD=ON",
            "-DNO_AVX2=ON",
        ],
        ":linux_x86_64": [
            "-DDYNAMIC_ARCH=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DUSE_OPENMP=ON",
            "-DUSE_THREAD=ON",
            "-DNO_AVX2=OFF",
        ],
        "//conditions:default": [],
    }),
    lib_source = ":all-src",
    out_data_dirs = ["share"],
    out_static_libs = select({
        ":macos_arm64": ["libopenblas.a"],
        ":linux_x86_64": ["libopenblas.a"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = ["@libomp"],
)
