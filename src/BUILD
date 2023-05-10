cc_library(
    name = "matrix",
    srcs = [
        "dm_sparse.c",
        "dm_dense.c",
        "dm_matrix.c",
        "dm_math.c",
        "dm_io.c",
        "dv_vector.c",
        "dv_math.c",
        "dm_special.c",
    ],
    hdrs = [
        "include/dm_matrix.h",
        "include/dm_math.h",
        "include/dm_io.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@dbg-macro",
         "@pbPlots",
    ],
)

