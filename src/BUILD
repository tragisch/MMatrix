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
        "//src:dm_matrix",
    ],
)
