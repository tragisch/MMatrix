# Using Apples Accelerate framework for BLAS
# https://developer.apple.com/documentation/accelerate/blas

cc_library(
    name = "accelerate",
    hdrs = ["$(GENDIR)/external/accelerate/Accelerate.tbd"],
    linkopts = [
        "-framework Accelerate",
    ],
    visibility = ["//visibility:public"],
)
