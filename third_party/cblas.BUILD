load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "cblas",
    srcs = glob([
        "lib/*.dylib",
    ]),
    hdrs = glob([
        "include/*.h",
    ]),
    includes = [
        "include"
    ],
    visibility = ["//visibility:public"],
)


