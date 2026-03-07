
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# homebrew installed local suitesparse
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

filegroup(
    name = "install_files",
    srcs = glob(["lib/*.a"]) + glob(["include/suitesparse/*.h"]),
    visibility = ["//visibility:public"],
)
