workspace(name = "DoubleMatrix")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

##########################
## C++ Toolchain
##########################

# http_archive(
#     name = "rules_cc",
#     urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.6/rules_cc-0.0.6.tar.gz"],
#     sha256 = "3d9e271e2876ba42e114c9b9bc51454e379cbf0ec9ef9d40e2ae4cec61a31b40",
#     strip_prefix = "rules_cc-0.0.6",
# )

# load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies", "rules_cc_toolchains")
# rules_cc_dependencies()
# rules_cc_toolchains()

##########################
## Foreign Build Tool Repositories
##########################

http_archive(
    name = "rules_foreign_cc",
    sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
    strip_prefix = "rules_foreign_cc-0.9.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.9.0.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

# This sets up some common toolchains for building targets. For more details, please see
# https://bazelbuild.github.io/rules_foreign_cc/0.9.0/flatten.html#rules_foreign_cc_dependencies
rules_foreign_cc_dependencies()

##########################
## Git Bazel Repositories
##########################
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "com_google_benchmark",
    remote = "https://github.com/google/benchmark.git",
    tag = "v1.8.0",
)

##########################
## Http Archives
##########################

# Simple Unit Testing for C
# https://github.com/ThrowTheSwitch/Unity/archive/v2.5.2.zip
# LICENSE: MIT
http_archive(
    name = "Unity",
    build_file = "@//:third_party/ThrowTheSwitch/Unity/BUILD",
    sha256 = "4598298723ecca1f242b8c540a253ae4ab591f6810cbde72f128961007683034",
    strip_prefix = "Unity-2.5.2",
    urls = [
        "https://github.com/ThrowTheSwitch/Unity/archive/refs/tags/v2.5.2.zip",
    ],
)

http_archive(
    name = "CMock",
    build_file = "@//:third_party/ThrowTheSwitch/CMock/BUILD",
    sha256 = "f342b8296aa934acfa3310a015938901e7df40ff7f5041c0ef3f5e6b13580207",
    strip_prefix = "CMock-2.5.3",
    url = "https://github.com/ThrowTheSwitch/CMock/archive/refs/tags/v2.5.3.zip",
)

# A few macros that prints and returns the value of a given expression
# for quick and dirty debugging, inspired by Rusts dbg!(…) macro and its C++ variant.
# https://github.com/eerimoq/dbg-macro
http_archive(
    name = "dbg-macro",
    build_file = "@//:third_party/dbg-macro/BUILD",
    sha256 = "2cd05a0ab0c93d115bf0ee476a5746189f3ced1d589abb098307daeaa57ef329",
    strip_prefix = "dbg-macro-0.12.1",
    url = "https://github.com/eerimoq/dbg-macro/archive/refs/tags/0.12.1.zip",
)


http_archive(
	name = "openblas",
	url = "http://github.com/xianyi/OpenBLAS/archive/v0.2.20.zip",
	sha256 = "bb5499049cf60b07274740a4ddd756daa0fe2c817d981d7fe7e5898dcf411fdc",
	strip_prefix = "OpenBLAS-0.2.20",
	build_file = "@//:third_party/openblas/BUILD"
)

##########################
## LOCAL REPOSITORIES (e.g. BREW)
##########################

# openblas
new_local_repository(
    name = "cblas",
    build_file = "./third_party/cblas/BUILD",
    path = "/opt/homebrew/Cellar/openblas/0.3.23",
)

new_local_repository(
    name = "gsl",
    build_file = "./third_party/gsl/BUILD",
    path = "/opt/homebrew/Cellar/gsl/2.7.1",
)

new_local_repository(
    name = "accelerate",
    build_file = "./third_party/accelerate/BUILD",
    path = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/",
)

local_repository(
    name = "fortran_rules",
    path = "third_party/openblas/tools/fortran_rules",
)



