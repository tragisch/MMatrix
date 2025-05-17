load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "pcg",
    srcs = glob(["src/*.c"]),
    hdrs = ["include/pcg_variants.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)