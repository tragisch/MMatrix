"""Root build file for MMatrix C project."""

load("//tools/install:def.bzl", "installer")

installer(
    name = "matrix_installer",
    data = [
        "//:LICENSE.txt",  # if available
        "//app/matrix",  # the target to be installed
        "//app/matrix:matrix_header",  # must be collected in a filegroup
        "@log",
        "@matio",
        "@openblas//:install_files",
        "@openmp//:libomp",
        "@pcg",
        "@suitesparse//:install_files",
        "@zlib",
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
