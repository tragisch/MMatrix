load("@rules_cc//cc:cc_library.bzl", "cc_library")

config_setting(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

genrule(
    name = "matioConfig_h",
    srcs = select({
        ":macos_arm64": ["@//third_party/matio:matioConfig.h"],
        ":linux_x86_64": ["@//third_party/matio:matioConfig-Linux.h"],
    }),
    outs = ["matioConfig.h"],
    cmd = "cp $< $@",
    visibility = ["//visibility:public"],
)

genrule(
    name = "matio_pubconf_h",
    srcs = select({
        ":macos_arm64": ["@//third_party/matio:matio_pubconf.h"],
        ":linux_x86_64": ["@//third_party/matio:matio_pubconf-Linux.h"],
    }),
    outs = ["matio_pubconf.h"],
    cmd = "cp $< $@",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "matio",
    srcs = glob(["src/*.c"]) + [
        "matioConfig.h",
        "matio_pubconf.h",
    ],
    hdrs = glob(["src/*.h"]),
    copts = [
        "-Wno-unused-but-set-variable",
    ],
    includes = [
        "src",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":matioConfig_h",
        ":matio_pubconf_h",
        "@zlib",
    ],
)
