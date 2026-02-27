"""Root build file for MMatrix C project."""

load("@phst_license_test//:def.bzl", "license_test")
load("//tools/install:def.bzl", "installer")

# identify missing license headers "bazel run //:license_test"
license_test(
    name = "license_test",
    timeout = "short",
    ignore = [".dir-locals.el"],
    marker = "//:MODULE.bazel",
    tags = ["manual"],
)

installer(
    name = "matrix_installer",
    data = [
        "//:LICENSE.txt",  # if available
        "//app/matrix",  # the target to be installed
        "//app/matrix:matrix_header",  # must be collected in a filegroup
    ],
    system_integration = False,  # set "True" symlink to /usr/local/lib and /usr/local/include
)

alias(
    name = "coverage_html",
    actual = "//tools:coverage_lcov_wrapper",
)

alias(
    name = "refresh_compile_commands",
    actual = "//tools:refresh_compile_commands",
)

alias(
    name = "refresh_compile_commands_raw",
    actual = "@wolfd_bazel_compile_commands//:generate_compile_commands",
)
