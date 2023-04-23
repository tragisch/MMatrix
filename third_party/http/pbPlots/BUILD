cc_library(
    name = "pbPlots",
    srcs = ["C/pbPlots.c",
            "C/supportLib.c"],
    hdrs = ["C/pbPlots.h",
            "C/supportLib.h"],
    includes = ["C"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "example3",
    srcs = ["example3.c"],
    deps = ["@pbPlots"],
    linkopts = [
        "-lm",
    ],
    visibility = ["//visibility:public"],
)