# Using Apples Accelerate framework for BLAS
# https://developer.apple.com/documentation/accelerate/blas

cc_library(
    name = "accelerate",
    hdrs = ["Current/Headers/accelerate.h"],
    linkopts = [
        "-framework Accelerate",
    ],
    includes = ["Current/Headers"],
    visibility = ["//visibility:public"],
)
