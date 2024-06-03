workspace(name = "DoubleMatrix")
# load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


git_repository(
    name = "com_google_benchmark",
    remote = "https://github.com/google/benchmark.git",
    tag = "v1.8.0",
)

# new_local_repository(
#     name = "accelerate",
#     build_file = "@//:third_party/accelerate.BUILD",
#     path = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/",
# )

new_local_repository(
    name = "sparsesuite",
    build_file = "@DoubleMatrix//third_party:sparsesuite.BUILD",
    path = "/opt/homebrew/Cellar/suite-sparse/7.7.0",
)



# # openblas
# new_local_repository(
#     name = "cblas",
#     build_file = "@//third_party/cblas/BUILD",
#     path = "/opt/homebrew/Cellar/openblas/0.3.23",
# )

# new_local_repository(
#     name = "gsl",
#     build_file = "@//third_party/gsl/BUILD",
#     path = "/opt/homebrew/Cellar/gsl/2.7.1",
# )

# Group the sources of the library so that CMake rule have access to it
# all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

# # OpenBLAS source code repository
# http_archive(
#    name = "openblas",
#    strip_prefix = "OpenBLAS-0.3.24",
#    build_file_content = all_content,
#    sha256 = "bb5499049cf60b07274740a4ddd756daa0fe2c817d981d7fe7e5898dcf411fdc",
#    url = "https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.24.tar.gz",
# )

# # Eigen source code repository
# http_archive(
#    name = "eigen",
#    build_file_content = all_content,
#    strip_prefix = "eigen-git-mirror-3.3.5",
#    urls = ["https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.tar.gz"],
# )






