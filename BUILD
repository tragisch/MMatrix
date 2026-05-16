"""Root build file for MMatrix C project."""

load("@mkdocs//:defs.bzl", "mkdocs_build", "mkdocs_config", "mkdocs_serve")
load("@rules_docs//docgen:defs.bzl", "docs")
load("//tools/install:def.bzl", "installer")

exports_files(["compile_commands.json"])

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

docs(
    name = "docs_src",
    srcs = glob(["doc/**/*.md"]) + [
        "//doc/examples:tensor_basic.md",
    ],
    readme_content = "MMatrix Dokumentation.",
)

mkdocs_config(
    name = "mkdocs_config",
    docs = ":docs_src",
    mkdocs_base = "doc/mkdocs.tpl.yaml",
)

mkdocs_build(
    name = "mkdocs",
    config = ":mkdocs_config",
    docs = [":docs_src"],
    site_dir = "site",
    visibility = ["//visibility:public"],
)

mkdocs_serve(
    name = "mkdocs.serve",
    config = ":mkdocs_config",
    docs = [":docs_src"],
    visibility = ["//visibility:public"],
)

alias(
    name = "docs",
    actual = ":mkdocs",
)

alias(
    name = "docs_serve",
    actual = ":mkdocs.serve",
)

alias(
    name = "docs_refresh_api",
    actual = "//tools:generate_api_docs",
)
