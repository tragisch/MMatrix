load("//third_party/http/ThrowTheSwitch/Unity:unity.bzl", "unity_test")

unity_test(
    size = "small",
    cexception = False,
    file_name = "test_DoubleMatrix.c",
    deps = [
        "//lib/misc",
        "//src:matrix"
    ],
)

unity_test(
    size = "small",
    cexception = False,
    file_name = "test_DoubleVector.c",
    deps = [
        "//lib/misc",
        "//src:matrix"
    ],
)

unity_test(
    size = "small",
    cexception = False,
    file_name = "test_dm_math.c",
    deps = [
        "//lib/misc",
        "//src:matrix",
        "@dbg-macro"
    ],
)

unity_test(
    size = "small",
    cexception = False,
    file_name = "test_dv_math.c",
    deps = [
        "//lib/misc",
        "//src:matrix",
        "@dbg-macro"
    ],
)