cc_library(
    name = "DoubleMatrix",
    srcs = [
        "matrix.c",
    ],
    hdrs = [
        "include/matrix.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//lib/misc",
        "//src:DoubleVector",
    ],
)

cc_library(
    name = "DoubleVector",
    srcs = [
        "vector.c",
    ],
    hdrs = [
        "include/vector.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//lib/misc",
    ],
)
