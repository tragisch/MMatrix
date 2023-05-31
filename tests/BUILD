load("//third_party/ThrowTheSwitch/Unity:unity.bzl", "unity_test", "generate_test_runner")

## Generate a test suite with all the tests in the current directory.

test_suite(
    name = "small_tests",
    tags = ["smoke"],
    visibility = ["//visibility:public"],
    
)

test_suite(
    name = "test_math",
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
    name = "test_my_config",
    srcs = [
        "test_DoubleMatrix.c",
    ],
    my_config = [
        "my_config.yml"
        ],
    deps = [
        "//src:matrix",
    ],
    tags = ["smoke"],
)


unity_test(
    name = "test_DenseMatrix",
    srcs = [
        "test_DenseMatrix.c",
    ],
    deps = [
        "//src:matrix",
    ],
    tags = ["smoke"],
)

unity_test(
    name = "test_DoubleVector",
    srcs = [
        "test_DoubleVector.c",
    ],
    deps = [
        "//src:matrix",
    ],
    tags = ["smoke"],
)
  

unity_test(
    name ="test_dm_math",
    srcs= [
        "test_dm_math.c",
    ],
    deps = [
        "//src:matrix",
    ],
    tags = ["math"],
)

unity_test(
    name ="test_dv_math",
    srcs= [
        "test_dv_math.c",
    ],
    deps = [
        "//src:matrix",
    ],
    tags = ["math"],
)
