cc_library(
    name = "CException",
    srcs = ["lib/CException.c"],
    hdrs = [
        "lib/CException.h",
    ],
    includes = ["lib"],
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        "@Unity",
    ],
)

cc_test(
    name = "test_exception",
    timeout = "short",
    srcs = ["test_exception.c"],
    visibility = ["//visibility:public"],
    deps = [
        "@CException",
        "@Unity",
    ],
)
