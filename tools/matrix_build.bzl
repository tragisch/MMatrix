"""This module provides a "helper2 to build a C binary for a matrix."""
load("@rules_cc//cc:defs.bzl", "cc_binary")

def matrix_cc_binary(name, srcs, deps = [], visibility = None):
    cc_binary(
        name = name,
        srcs = srcs,
        deps = deps + [
            "//src:matrix"
        ],
        visibility = visibility,    )

