
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "suitesparse",
    srcs = glob([
        "lib/*.a",
    ]),
    hdrs = glob([
        "include/suitesparse/*.h",
    ]),
    includes = [
        "include/suitesparse",
    ],
    visibility = ["//visibility:public"],
)