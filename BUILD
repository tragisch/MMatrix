cc_library(
    name = "DoubleMatrix",
    srcs = ["src/dm.c", "src/dms.c"],
    hdrs = ["src/include/dm.h", 
    "src/include/dms.h"],
    includes = ["src/include"],
    copts = ["-DACCELERATE_NEW_LAPACK"],
    visibility = ["//visibility:public"],
    deps = ["@sparsesuite"],
)

cc_library(
    name = "dm",
    srcs = ["src/dm.c"],
    hdrs = ["src/include/dm.h"],
    includes = ["src/include"],
    copts = ["-DACCELERATE_NEW_LAPACK"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "dms",
    srcs = ["src/dms.c"],
    hdrs = ["src/include/dms.h"],
    includes = ["src/include"],
    visibility = ["//visibility:public"],
    deps = ["@sparsesuite",],
)