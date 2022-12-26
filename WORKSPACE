load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Simple Unit Testing for C
# https://github.com/ThrowTheSwitch/Unity/archive/v2.5.2.zip
# LICENSE: MIT
http_archive(
    name = "Unity",
    build_file = "@//:packages/http/ThrowTheSwitch/Unity/BUILD",
    sha256 = "4598298723ecca1f242b8c540a253ae4ab591f6810cbde72f128961007683034",
    strip_prefix = "Unity-2.5.2",
    urls = [
        "https://github.com/ThrowTheSwitch/Unity/archive/refs/tags/v2.5.2.zip",
    ],
)

http_archive(
    name = "CException",
    build_file = "@//:packages/http/ThrowTheSwitch/CException/BUILD",
    sha256 = "fd43ca698f86f75805fc9440d814dfa11c3e72674a52bfda43344f4371e8bcd9",
    strip_prefix = "CException-1.3.3",
    url = "https://github.com/ThrowTheSwitch/CException/archive/refs/tags/v1.3.3.zip",
)

http_archive(
    name = "CMock",
    build_file = "@//:packages/http/ThrowTheSwitch/CMock/BUILD",
    sha256 = "f342b8296aa934acfa3310a015938901e7df40ff7f5041c0ef3f5e6b13580207",
    strip_prefix = "CMock-2.5.3",
    url = "https://github.com/ThrowTheSwitch/CMock/archive/refs/tags/v2.5.3.zip",
)

########## LOCAL REPOSITORIES (e.g. BREW)

# stellt Standdard Datenstrukturen zur Verfügung (mit Brew installiert.)
new_local_repository(
    name = "glib",
    build_file = "./packages/brew/gnu-lib/glib.BUILD",
    path = "/opt/homebrew/Cellar/glib/2.72.1",
)
