load("//:tools/install_library.bzl", "install_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "DoubleMatrix",
    srcs = [
        "src/dm.c",
        "src/dms.c"
    ],
    hdrs = [
        "src/include/dm.h", 
        "src/include/dms.h"
    ],
    includes = ["src/include"],
   
    copts = [
        "-DACCELERATE_NEW_LAPACK",
    ],
    linkopts = [
        "-framework Accelerate",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@sparsesuite//:sparsesuite", 
        "@matio//:matio"
    ],
)

cc_library(
    name = "dm",
    srcs = ["src/dm.c"],
    hdrs = ["src/include/dm.h"],
    includes = ["src/include"],
    copts = ["-DACCELERATE_NEW_LAPACK"],
    linkopts = [
        "-framework Accelerate",
    ],
    visibility = ["//visibility:public"],
    deps = ["@matio//:matio"],
)

cc_library(
    name = "dms",
    srcs = ["src/dms.c"],
    hdrs = ["src/include/dms.h"],
    includes = ["src/include"],
    visibility = ["//visibility:public"],
    deps = ["@sparsesuite",],
)

install_library(
    name = "install",
    srcs = [":DoubleMatrix"],  # Reference the library target
    hdrs = ["dms.h", "dm.h"],  # Explicitly list headers to include
    target_dir = "/Users/uwe/",  # Specify your output directory
)