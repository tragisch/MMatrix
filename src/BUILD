cc_library(
    name = "matrix",
    srcs = [
        "dm_matrix.c",
        "dm_math.c",
        "dm_manipulate.c",
        "dm_io.c",
        "dv_vector.c",
        "dv_math.c",
        "dm_special.c",
    ],
    hdrs = [
        "include/dm.h",
        "include/dm_internals.h",
        "include/dv_vector.h",
        "include/dm_math.h",
        "include/dm_io.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@dbg-macro",
        "//third_party/source/klib:klib",
    ],
)

