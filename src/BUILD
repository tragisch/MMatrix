load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "DoubleMatrix",
    srcs = [
        "dm.c",
        "dm_convert.c",
        "dm_io.c",
        "dms.c",
    ],
    hdrs = [
        "include/dm.h",
        "include/dm_convert.h",
        "include/dm_io.h",
        "include/dms.h",
    ],
    copts = [
        "-DACCELERATE_NEW_LAPACK",
        "-DACCELERATE_LAPACK_ILP64",
    ],
    includes = ["include"],
    linkopts = [
        "-framework Accelerate",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@matio",
        "@suitesparse",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.c"],
    deps = [":DoubleMatrix"],
)
