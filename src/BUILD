cc_library(
    name = "dm",
    srcs = ["dm.c"],
    hdrs = ["include/dm.h"],
    includes = ["include"],
    copts = ["-DACCELERATE_NEW_LAPACK"],
    visibility = ["//visibility:public"],
    deps = ["@accelerate"],
)

cc_library(
    name = "dms",
    srcs = ["dms.c"],
    hdrs = ["include/dms.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = ["@sparsesuite", "@dbg-macro",],
)

