cc_library(
    name = "CException",
    srcs = ["lib/CException.c"],
    hdrs = [
        "lib/CException.h",
    ],
    includes = ["lib"],
    visibility = ["//visibility:public"],
    deps = [
        "@Unity",
    ],
)

cc_test(
    name = "TestException",
    srcs = ["TestException.c"],
    visibility = ["//visibility:public"],
    deps = [
        "@CException",
        "@Unity",
    ],
)
