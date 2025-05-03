load("@phst_license_test//:def.bzl", "license_test")
load("//tools:install_library.bzl", "install_library_macro")

# identify missing license headers "bazel run //:license_test"
license_test(
    name = "license_test",
    timeout = "short",
    ignore = [".dir-locals.el"],
    marker = "//:MODULE.bazel",
)

filegroup(
    name = "all_headers",
    srcs = glob(["src/include/*.h"]),
    visibility = ["//visibility:public"],
)

# install library macro
install_library_macro(
    name = "install_my_lib",
    srcs = ["//src:matrix"],
    hdrs = ["//src/include:all_headers"],
)