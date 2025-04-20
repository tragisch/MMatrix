load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "libomp",
    srcs = [
        "lib/libomp.dylib",
    ],
    hdrs = [
        "include/omp.h",
        "include/omp-tools.h",
        "include/ompt.h",
    ],
    includes = [
        "include",
    ],
    visibility = [
        "//visibility:public",
    ],
)