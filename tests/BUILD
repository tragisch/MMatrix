load("//third_party/ThrowTheSwitch/Unity:unity.bzl", "unity_test")


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


unity_test(
    name = "test_DoubleMatrix",
    srcs = [
        "test_DoubleMatrix.c",
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
