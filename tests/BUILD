load("//third_party/ThrowTheSwitch/Unity:unity.bzl", "unity_test")

## Generate a test suite with all the tests in the current directory.

test_suite(
    name = "smoke",
    tags = ["smoke"],
    visibility = ["//visibility:public"],
    
)

test_suite(
    name = "math",
    tags = ["math"],
    visibility = ["//visibility:public"],
)

## Unit tests:

unity_test(
    name = "test_dm_create",
    srcs = [
        "test_dm_create.c",
    ],
    deps = [
        "//src:matrix",
    ],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_create_special",
    srcs = [
        "test_dm_create_special.c",
    ],
    deps = [
        "//src:matrix",
    ],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_create",
    srcs = ["test_dm_create.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)


unity_test(
    name = "test_dm_math",
    srcs = ["test_dm_math.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["math"],
)

unity_test(
    name = "test_dm_math_vector",
    srcs = ["test_dm_math_vector.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["math"],
)

unity_test(
    name = "test_dm_vector",
    srcs = ["test_dm_vector.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_getset",
    srcs = ["test_dm_getset.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_create_special",
    srcs = ["test_dm_create_special.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_modify",
    srcs = ["test_dm_modify.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_modify_insert",
    srcs = ["test_dm_modify_insert.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_modify_remove",
    srcs = ["test_dm_modify_remove.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_modify_resize",
    srcs = ["test_dm_modify_resize.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_modify_reshape",
    srcs = ["test_dm_modify_reshape.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_modify_cleanup",
    srcs = ["test_dm_modify_cleanup.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)


unity_test(
    name = "test_dm_format_market",
    srcs = ["test_dm_format_market.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dm_realloc",
    srcs = ["test_dm_realloc.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)




