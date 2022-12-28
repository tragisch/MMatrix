cc_library(
    name = "misc",
    srcs = ["misc.c"],
    hdrs = ["misc.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "example_misc",
    srcs = ["example_misc.c"],
    visibility = ["//visibility:public"],
    deps = ["//lib/misc"],
)
