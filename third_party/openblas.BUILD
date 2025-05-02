load("@rules_cc//cc:cc_library.bzl", "cc_library")

# homebrew installed local openblas
cc_library(
    name = "openblas",
    srcs = ["lib/libopenblas.a"],
    hdrs = glob([
        "include/*.h",
    ]),
    includes = [
        "include"
    ],
    visibility = ["//visibility:public"],
)


