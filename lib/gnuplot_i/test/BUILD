load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "anim",
    srcs = ["anim.c"],
    deps = [
        "//lib/gnuplot_i/src:gnuplot_i",
    ],
)

cc_binary(
    name = "example",
    srcs = [
        "example.c",
    ],
    deps = [
        "//lib/gnuplot_i/src:gnuplot_i",
    ],
)

cc_binary(
    name = "sinepng",
    srcs = [
        "sinepng.c",
    ],
    deps = [
        "//lib/gnuplot_i/src:gnuplot_i",
    ],
)
