load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
    name = "CException",
    build_file = "@//:third_party/http/ThrowTheSwitch/CException/BUILD",
    sha256 = "fd43ca698f86f75805fc9440d814dfa11c3e72674a52bfda43344f4371e8bcd9",
    strip_prefix = "CException-1.3.3",
    url = "https://github.com/ThrowTheSwitch/CException/archive/refs/tags/v1.3.3.zip",
)

http_archive(
    name = "CMock",
    build_file = "@//:third_party/http/ThrowTheSwitch/CMock/BUILD",
    sha256 = "f342b8296aa934acfa3310a015938901e7df40ff7f5041c0ef3f5e6b13580207",
    strip_prefix = "CMock-2.5.3",
    url = "https://github.com/ThrowTheSwitch/CMock/archive/refs/tags/v2.5.3.zip",
)

# fff is a micro-framework for creating fake C functions for tests
# https://github.com/meekrosoft/fff
http_archive(
    name = "fff",
    build_file = "@//:third_party/http/fff/BUILD",
    sha256 = "510efb70ab17a0035affd170960401921c9cc36ec81002ed00d2bfec6e08f385",
    strip_prefix = "fff-1.1",
    url = "https://github.com/meekrosoft/fff/archive/refs/tags/v1.1.tar.gz",
)

# gnuplot is a freely available, command-driven graphical display tool for Unix.
# http://ndevilla.free.fr/gnuplot/
http_archive(
    name = "gnuplot_i",
    build_file = "@//:third_party/http/gnuplot_i/BUILD",
    sha256 = "be453cf2683353c9330d7784ffc2cae5f58a5c864aa1adb09baedf3e2caf5e3a",
    strip_prefix = "gnuplot_i-master",
    url = "https://github.com/mithodin/gnuplot_i/archive/refs/heads/master.zip",
)

########## LOCAL REPOSITORIES (e.g. BREW)

# stellt Standdard Datenstrukturen zur Verfügung (mit Brew installiert.)
new_local_repository(
    name = "glib",
    build_file = "./third_party/brew/gnu-lib/glib.BUILD",
    path = "/opt/homebrew/Cellar/glib/2.74.0",
)
