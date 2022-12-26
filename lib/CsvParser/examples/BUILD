load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "csv_with_header",
    srcs = [
        "usage_example_with_header.c",
    ],
    copts = ["-Ideps/CsvParser/src"],
    deps = [
        "//deps/CsvParser/src:csvparser",
    ],
)

cc_binary(
    name = "csv_simple",
    srcs = [
        "usage_example.c",
    ],
    copts = ["-Ideps/CsvParser/src"],
    deps = [
        "//deps/CsvParser/src:csvparser",
    ],
)
