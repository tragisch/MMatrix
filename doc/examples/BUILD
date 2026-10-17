load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

exports_files([
    "tensor_basic.md",
])

cc_binary(
    name = "tensor_basic_example",
    srcs = ["tensor_basic_example.c"],
    deps = ["//app/tensor:st"],
)
