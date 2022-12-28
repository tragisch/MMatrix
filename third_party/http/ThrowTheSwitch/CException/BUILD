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
    name = "test_exception",
    srcs = ["test_exception.c"],
    visibility = ["//visibility:public"],
    deps = [
        "@CException",
    ],
)
