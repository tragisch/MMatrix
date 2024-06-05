workspace(name = "DoubleMatrix")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "com_google_benchmark",
    remote = "https://github.com/google/benchmark.git",
    tag = "v1.8.0",
)

new_local_repository(
    name = "sparsesuite",
    build_file = "@DoubleMatrix//third_party:sparsesuite.BUILD",
    path = "/opt/homebrew/Cellar/suite-sparse/7.7.0",
)

http_archive(
    name = "zlib",
    sha256 = "ff0ba4c292013dbc27530b3a81e1f9a813cd39de01ca5e0f8bf355702efa593e",
    build_file = "@//:third_party/zlib.BUILD",
    strip_prefix = "zlib-1.3",
    urls = [
        "https://github.com/madler/zlib/releases/download/v1.3/zlib-1.3.tar.gz",
    ],
)

http_archive(
    name = "matio",
    sha256 = "0a6aa00b18c4512b63a8d27906b079c8c6ed41d4b2844f7a4ae598e18d22d3b3",
    build_file = "@//:third_party/matio.BUILD",
    strip_prefix = "matio-1.5.27",
    urls = [
        "https://github.com/tbeu/matio/releases/download/v1.5.27/matio-1.5.27.tar.gz",
    ],
)

local_repository(
    name = "hdf5",
    path = "lib/hdf5",
)


# # openblas
# new_local_repository(
#     name = "cblas",
#     build_file = "@//:third_party/cblas.BUILD",
#     path = "/opt/homebrew/Cellar/openblas/0.3.23",
# )

# new_local_repository(
#     name = "gsl",
#     build_file = "@//:third_party/gsl.BUILD",
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










