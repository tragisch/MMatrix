load("//third_party/http/ThrowTheSwitch/Unity:unity.bzl", "unity_test")

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

unity_test(
    size = "small",
    cexception = False,
    file_name = "test_misc.c",
    deps = [
        "//lib/misc",
    ],
)
