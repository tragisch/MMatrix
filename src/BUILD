cc_binary(
    name = "main",
    srcs = ["main.c"],
    visibility = ["//visibility:public"],
    deps = [
        "//src:all_libs",
    ],
)

cc_library(
    name = "all_libs",
    srcs = glob(["*.c"]),
    hdrs = glob(["**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hello",
    srcs = ["hello.c"],
    hdrs = ["include/hello.h"],
    visibility = ["//visibility:public"],
)
