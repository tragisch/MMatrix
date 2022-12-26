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
        "//src:DoubleVector",
        "//src:pprint",
        "//src:utils",
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
        "//src:pprint",
        "//src:utils",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.c",
    ],
    hdrs = [
        "include/utils.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "pprint",
    srcs = [
        "pprint2D.c",
    ],
    hdrs = [
        "include/pprint2D.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:private"],
    deps = ["//src:utils"],
)
