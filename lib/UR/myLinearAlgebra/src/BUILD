cc_library(
    name = "matrix",
    srcs = ["matrix.c"],
    hdrs = ["matrix.h"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "example_matrix",
    srcs = ["example_matrix.c"],
    deps = ["//lib/UR/myLinearAlgebra/src:matrix"],
)
