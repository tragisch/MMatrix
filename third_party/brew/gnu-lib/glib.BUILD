cc_library(
    name = "glib",
    srcs = glob([
        "**/*.dylib",
    ]),
    hdrs = glob([
        "include/glib-2.0/*.h",
        "include/glib-2.0/**/*.h",
        "lib/glib-2.0/include/*.h",
    ]),
    includes = [
        "include/gio-unix-2.0/gio",
        "include/glib-2.0",
        "include/glib-2.0/gio",
        "include/glib-2.0/glib",
        "include/glib-2.0/gobject",
        "lib/glib-2.0/include",
    ],
    visibility = ["//visibility:public"],
)
