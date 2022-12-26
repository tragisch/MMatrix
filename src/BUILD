cc_library(
    name = "DoubleMatrix",
    srcs = [
        "matrix.c",
        "pprint2Dmatrix.c",
    ],
    hdrs = [
        "include/matrix.h",
        "include/pprint2Dmatrix.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "example_matrix",
    srcs = ["example_matrix.c"],
    deps = ["//lib/UR/DoubleMatrix"],
)
