load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "cs_demo2",
    srcs = ["cs_demo2.c"],
    visibility = ["//visibility:public"],
    deps = [
        "cs_demo",
        "//third_party/csparse:CSparse",
    ],
)

cc_library(
    name = "cs_demo",
    srcs = [
        "cs_demo.c",
        "cs_demo.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/csparse:CSparse",
    ],
)
