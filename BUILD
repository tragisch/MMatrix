load("@phst_license_test//:def.bzl", "license_test")
load("//tools/bazel/install:def.bzl", "installer")

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
        "//src:matrix",        # the target to be installed
        "//src:matrix_header", # must be collected in a filegroup
        "//:LICENSE.txt",      # if available
        ],
    system_integration = False, # set "True" symlink to /usr/local/lib and /usr/local/include
)