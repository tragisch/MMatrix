cc_library(
    name = "matrix",
    srcs = glob([
        "dm_*.c",
        "dv_*.c",
    ]),
    hdrs = glob([
        "include/*.h",
    ]),
    includes = ["include"],
    copts = select({
        ":accelerate": ["-DACCELERATE_NEW_LAPACK"],
        ":blas": [],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [] + select({
        ":accelerate": [":apple_accelerate", "@dbg-macro", "//third_party/jburkardt:c_src"],
        ":blas": ["@cblas", "@dbg-macro", "//third_party/jburkardt:c_src"],
        "//conditions:default": ["@cblas","@dbg-macro", "//third_party/jburkardt:c_src"],
    }),
)

config_setting(
    name = "accelerate",
    values = {
        "define": "accelerate=yes"
    }
)
config_setting(
    name = "blas",
    values = {
        "define": "blas=yes"
    }
)

config_setting(
    name = "macos",
    values = {"cpu": "darwin_arm64"},
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)


cc_library(
    name = "apple_accelerate",
    hdrs = glob(["src/include/*.h"]),
    linkopts = [
        "-framework Accelerate",
    ],
    visibility = ["//visibility:public"],
)

