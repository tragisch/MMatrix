load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])  # BSD/MIT-like license (for zlib)

cc_library(
    name = "zlib",
    srcs = [
        "adler32.c",
        "compress.c",
        "crc32.c",
        "crc32.h",
        "deflate.c",
        "deflate.h",
        "gzclose.c",
        "gzguts.h",
        "gzlib.c",
        "gzread.c",
        "gzwrite.c",
        "infback.c",
        "inffast.c",
        "inffast.h",
        "inffixed.h",
        "inflate.c",
        "inflate.h",
        "inftrees.c",
        "inftrees.h",
        "trees.c",
        "trees.h",
        "uncompr.c",
        "zconf.h",
        "zutil.c",
        "zutil.h",
    ],
    hdrs = ["zlib.h"],
    copts = [
        "-O",
        "-Wno-implicit-function-declaration",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)

# /* zlib.h -- interface of the 'zlib' general purpose compression library
#   version 1.3, August 18th, 2023

#   Copyright (C) 1995-2023 Jean-loup Gailly and Mark Adler

#   This software is provided 'as-is', without any express or implied
#   warranty.  In no event will the authors be held liable for any damages
#   arising from the use of this software.

#   Permission is granted to anyone to use this software for any purpose,
#   including commercial applications, and to alter it and redistribute it
#   freely, subject to the following restrictions:

#   1. The origin of this software must not be misrepresented; you must not
#      claim that you wrote the original software. If you use this software
#      in a product, an acknowledgment in the product documentation would be
#      appreciated but is not required.
#   2. Altered source versions must be plainly marked as such, and must not be
#      misrepresented as being the original software.
#   3. This notice may not be removed or altered from any source distribution.

#   Jean-loup Gailly        Mark Adler
#   jloup@gzip.org          madler@alumni.caltech.edu

# */
