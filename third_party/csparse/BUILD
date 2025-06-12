load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "CSparse",
    srcs = glob(["Source/*.c"]),
    hdrs = [
        "Include/cs.h",
    ],
    includes = ["Include"],
    visibility = ["//visibility:public"],
)