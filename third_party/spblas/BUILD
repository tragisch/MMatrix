load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "spblas",
    srcs = glob([
        "*.c",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    includes = [
        "."
    ],
    deps = [
        "@openblas//:openblas",
    ],
    visibility = ["//visibility:public"],
)