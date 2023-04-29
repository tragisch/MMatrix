cc_library(
    name = "matrix",
    srcs = [
        "sp_coo_matrix.c",
        "sp_csc_matrix.c",
        "sp_csr_matrix.c",
        "dm_matrix.c",
        "dm_math.c",
        "dm_io.c",
        "dv_vector.c",
        "dv_math.c",
        "sp_math.c",
        "sp_special.c",
    ],
    hdrs = [
        "include/dm_matrix.h",
        "include/dm_math.h",
        "include/dm_io.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//lib/misc",
        "@dbg-macro",
         "@pbPlots",
    ],
)


cc_library(
    name = "sp_matrix",
    srcs = [
        "sp_matrix.c",
    ],
    hdrs = [
        "include/sp_matrix.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//lib/misc",
        "@dbg-macro",
    ],
)

cc_library(
    name = "dm_matrix",
    srcs = [
        "dm_matrix.c",
    ],
    hdrs = [
        "include/dm_matrix.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//lib/misc",
        "//src:sp_matrix",
        "@dbg-macro",
    ],
)

cc_library(
    name = "dm_math",
    srcs = [
        "dm_math.c",
    ],
    hdrs = [
        "include/dm_math.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//lib/misc",
        "//src:sp_matrix",
        "//src:dm_matrix",
    ],
)

cc_library(
    name = "dm_io",
    srcs = [
        "dm_io.c",
    ],
    hdrs = [
        "include/dm_io.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//lib/misc",
        "//src:sp_matrix",
        "//src:dm_matrix",
        "//src:dm_math",
    ],
)
