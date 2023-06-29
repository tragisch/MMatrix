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
        ":macos": ["-DACCELERATE_NEW_LAPACK"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@dbg-macro",
       # "@cblas",
       "@accelerate",
        "//third_party/jburkardt:c_src",
    ],
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
    name = "accelerate",
    hdrs = glob(["src/include/*.h"]),
    linkopts = [
        "-framework Accelerate",
    ],
    visibility = ["//visibility:public"],
)

