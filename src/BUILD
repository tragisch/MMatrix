cc_library(
    name = "matrix",
    srcs = glob([
        "dm_*.c",
        "dv_*.c",
    ]),
    hdrs = glob([
        "include/*.h",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@dbg-macro",
        "@cblas",
    ],
)

