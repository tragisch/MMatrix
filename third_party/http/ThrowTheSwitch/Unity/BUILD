# load("//third_party/http/ThrowTheSwitch/Unity:generator.bzl", "file_generator")

cc_library(
    name = "Unity",
    srcs = ["src/unity.c"],
    hdrs = [
        "src/unity.h",
        "src/unity_internals.h",
    ],
    copts = ["-DUNITY_INCLUDE_DOUBLE"],
    includes = ["src"],
    visibility =  ["//visibility:public"],
)

filegroup(
    name = "TestRunnerGenerator",
    srcs = ["auto/generate_test_runner.rb"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "HelperScripts",
    srcs = glob(["auto/*.rb"]),
    visibility = ["//visibility:public"],
)

# package_group(
#     name = "Friends",
#     packages = [
#         "//third_party/http/ThrowTheSwitch/CException",
#         "//third_party/http/ThrowTheSwitch/Unity",
#         "//third_party/http/ThrowTheSwitch/CMock",
#     ],
# )


# file_generator(
#     name = "file",
#     arg = "some_arg",
# )

# file_generator(
#     name = "file-two",
#     arg = "some_arg_two",
# )

# file_generator(
#     name = "file-three",
#     arg = "some_arg_three",
# )
