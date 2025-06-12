load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "mm_offloading",
    srcs = [
        "mm_hybrid.c",
        "mm_utils.c",
    ],
    hdrs = [
        "mm_hybrid.c",
        "mm_utils.c",
    ],
    includes = [
        ".",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@libomp",
    ],
)
