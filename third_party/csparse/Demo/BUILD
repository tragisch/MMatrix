
cc_binary(
    name = "cs_demo2",
    srcs = ["cs_demo2.c"],
    deps = [
           "//third_party/csparse:CSparse",
           "cs_demo"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cs_demo",
    srcs = ["cs_demo.c","cs_demo.h"],
    deps = [
          "//third_party/csparse:CSparse"
    ],
    visibility = ["//visibility:public"],
)