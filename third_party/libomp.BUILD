load("@rules_cc//cc:cc_library.bzl", "cc_library")

# homebrew installed local libomp
cc_library(
    name = "libomp",
    srcs = [
        "lib/libomp.a",
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