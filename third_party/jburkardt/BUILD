load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "c_src",
    srcs = glob([
        "*.c",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    includes = [
        "."
    ],
    visibility = ["//visibility:public"],
)



