load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

sh_binary(
    name = "coverage_lcov_wrapper",
    srcs = ["coverage_lcov_wrapper.sh"],
    data = ["@bazel_tools//tools/test:coverage_report_generator"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "refresh_compile_commands",
    srcs = ["refresh_compile_commands.sh"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "coverage",
    values = {"collect_code_coverage": "true"},
    visibility = ["//visibility:public"],
)
