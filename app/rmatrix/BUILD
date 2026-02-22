load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

rust_library(
    name = "rmatrix",
    srcs = [
        "src/ffi.rs",
        "src/lib.rs",
    ],
    crate_root = "src/lib.rs",
    edition = "2021",
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = ["//app/matrix:sm"],
)

rust_test(
    name = "rmatrix_tests",
    crate = ":rmatrix",
    tags = ["manual"],
)
