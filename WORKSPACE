load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

##########################
## Hedron's Compile Commands Extractor for Bazel
## https://github.com/hedronvision/bazel-compile-commands-extractor
##########################

# http_archive(
#     name = "hedron_compile_commands",
#     strip_prefix = "bazel-compile-commands-extractor-5bb5ff2f32d542a986033102af771aa4206387b9",
#     # 18594913275f4b70343d5003d03d8e1b
#     # Replace the commit hash in both places (below) with the latest, rather than using the stale one here.
#     # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
#     url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/5bb5ff2f32d542a986033102af771aa4206387b9.tar.gz",

#     # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
# )

# load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

# hedron_compile_commands_setup()

##########################
## Code Coverage
##########################
# How to use:
# 1. Generate coverage data with bazel coverage //your/targets/... --instrumentation_filter=<...>
# 2. Build the coverage report generator: bazel build @hchauvin_bazel_coverage_report//report:bin
# 3. Generate the report: bazel-bin/external/hchauvin_bazel_coverage_report/report/bin --dest_dir=<dest dir>

# git_repository(
#     name = "hchauvin_bazel_coverage_report",
#     remote = "https://github.com/hchauvin/bazel-coverage-report.git",
#     commit = "{HEAD}",
# )
# load("@hchauvin_bazel_coverage_report//report:defs.bzl", "bazel_coverage_report_repositories")
# bazel_coverage_report_repositories()  # lcov, ...


##########################
## Http Archives
##########################

# Simple Unit Testing for C
# https://github.com/ThrowTheSwitch/Unity/archive/v2.5.2.zip
# LICENSE: MIT
http_archive(
    name = "Unity",
    build_file = "@//:third_party/http/ThrowTheSwitch/Unity/BUILD",
    sha256 = "4598298723ecca1f242b8c540a253ae4ab591f6810cbde72f128961007683034",
    strip_prefix = "Unity-2.5.2",
    urls = [
        "https://github.com/ThrowTheSwitch/Unity/archive/refs/tags/v2.5.2.zip",
    ],
)

http_archive(
    name = "CMock",
    build_file = "@//:third_party/http/ThrowTheSwitch/CMock/BUILD",
    sha256 = "f342b8296aa934acfa3310a015938901e7df40ff7f5041c0ef3f5e6b13580207",
    strip_prefix = "CMock-2.5.3",
    url = "https://github.com/ThrowTheSwitch/CMock/archive/refs/tags/v2.5.3.zip",
)

# A few macros that prints and returns the value of a given expression
# for quick and dirty debugging, inspired by Rusts dbg!(…) macro and its C++ variant.
# https://github.com/eerimoq/dbg-macro
http_archive(
    name = "dbg-macro",
    build_file = "@//:third_party/http/dbg-macro/BUILD",
    sha256 = "2cd05a0ab0c93d115bf0ee476a5746189f3ced1d589abb098307daeaa57ef329",
    strip_prefix = "dbg-macro-0.12.1",
    url = "https://github.com/eerimoq/dbg-macro/archive/refs/tags/0.12.1.zip",
)

