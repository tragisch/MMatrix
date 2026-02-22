load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("@phst_license_test//:def.bzl", "license_test")
load("//tools/install:def.bzl", "installer")

# identify missing license headers "bazel run //:license_test"
license_test(
    name = "license_test",
    timeout = "short",
    ignore = [".dir-locals.el"],
    marker = "//:MODULE.bazel",
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

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = {
        "//app/matrix": "--define=USE_ACCELERATE_MPS=1",
    },
)
