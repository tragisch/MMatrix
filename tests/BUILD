load("//:tools/unity_test.bzl", "unity_test")

## Unit tests:

unity_test(
    name = "test_dm",
    srcs = ["test_dm.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:DoubleMatrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dms",
    srcs = ["test_dms.c",],
    my_config = ["my_config.yml"],
     deps = ["//src:DoubleMatrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_convert",
    srcs = ["test_convert.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:DoubleMatrix",],
    tags = ["smoke"],
)

