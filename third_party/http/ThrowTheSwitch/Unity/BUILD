#load("//third_party/http/ThrowTheSwitch/Unity:generator.bzl", "file_generator")

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


genrule(
    name = "envinfo",
    srcs = [
        "a.txt",
        "b.txt",
    ],
    outs = ["envinfo2.txt"],
    cmd_bash = """(
        echo "-----"
        pwd
        ls -l
        echo "-----"
        echo "whoami     = $$(whoami)"
        echo "TARGET_CPU = $(TARGET_CPU)"
        echo "BINDIR     = $(BINDIR)"
        echo "GENDIR     = $(GENDIR)"
        echo "OUTS       = $(OUTS)"
        echo "SRCS       = $(SRCS)"
        echo "RULEDIR    = $(RULEDIR)"
        echo "a.txt      = $(location :a.txt)"
        echo "b.txt      = $(location :b.txt)"
        ) | tee $@""",
)


# file_generator(
#     name = "file",
# )

# file_generator(
#     name = "file-two",
# )

# file_generator(
#     name = "file-three",
# )
