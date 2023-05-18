cc_library(
    name = "dbg-macro",
    hdrs = ["include/dbg.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "example_dbg-macro",
    srcs = [
        "example_dbg-macro.c",
    ],
    deps = [
        "@dbg-macro",
    ],
)
