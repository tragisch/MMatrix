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

    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = {
        "//app/matrix": "--define=USE_ACCELERATE_MPS=1",
    },
    # No need to add flags already in .bazelrc. They're automatically picked up.
    # If you don't need flags, a list of targets is also okay, as is a single target string.
    # Wildcard patterns, like //... for everything, *are* allowed here, just like a build.
    # As are additional targets (+) and subtractions (-), like in bazel query https://docs.bazel.build/versions/main/query.html#expressions
    # And if you're working on a header-only library, specify a test or binary target that compiles it.
)
