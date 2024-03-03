workspace(name = "DoubleMatrix")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


##########################
## Git Bazel Repositories
##########################

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


# http_archive(
# 	name = "openblas",
# 	url = "http://github.com/xianyi/OpenBLAS/archive/v0.2.20.zip",
# 	sha256 = "bb5499049cf60b07274740a4ddd756daa0fe2c817d981d7fe7e5898dcf411fdc",
# 	strip_prefix = "OpenBLAS-0.2.20",
# 	build_file = "@//:third_party/openblas/BUILD"
# )


# Group the sources of the library so that CMake rule have access to it
all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

# OpenBLAS source code repository
http_archive(
   name = "openblas",
   strip_prefix = "OpenBLAS-0.3.24",
   build_file_content = all_content,
   sha256 = "bb5499049cf60b07274740a4ddd756daa0fe2c817d981d7fe7e5898dcf411fdc",
   url = "https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.24.tar.gz",
)

# Eigen source code repository
http_archive(
   name = "eigen",
   build_file_content = all_content,
   strip_prefix = "eigen-git-mirror-3.3.5",
   urls = ["https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.tar.gz"],
)

##########################
## NON BAZEL WAY: USING LOCAL COMPILED REPOSITORIES (e.g. BREW)
##########################

# openblas
new_local_repository(
    name = "cblas",
    build_file = "@//third_party/cblas/BUILD",
    path = "/opt/homebrew/Cellar/openblas/0.3.23",
)

new_local_repository(
    name = "gsl",
    build_file = "@//third_party/gsl/BUILD",
    path = "/opt/homebrew/Cellar/gsl/2.7.1",
)

new_local_repository(
    name = "accelerate",
    build_file = "@//third_party/accelerate/BUILD",
    path = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/",
)

local_repository(
    name = "fortran_rules",
    path = "third_party/openblas/tools/fortran_rules",
)


# # OpenBLAS source code repository
# http_archive(
#    name = "openblas",
#    build_file_content = all_content,
#    strip_prefix = "OpenBLAS-0.3.2",
#    https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.24.tar.gz
#    urls = ["https://github.com/xianyi/OpenBLAS/archive/v0.3.2.tar.gz"],
# )






