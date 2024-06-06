load("//:tools/unity_test.bzl", "unity_test")

## Unit tests:

unity_test(
    name = "test_dm",
    srcs = ["test_dm.c",],
    my_config = ["my_config.yml"],
    deps = ["//:dm",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dms",
    srcs = ["test_dms.c",],
    my_config = ["my_config.yml"],
    deps = ["//:dms",],
    tags = ["smoke"],
)

