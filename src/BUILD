load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")


# build options for different architectures
dm_copts = [
    "-O2",
    "-fno-fast-math",
    "-Xpreprocessor",
]

sm_copts = [
    "-O3",
    "-ffast-math",
    "-funsafe-math-optimizations",
    "-Xpreprocessor",
]

default_deps = [
    "@matio",
]

# matrix library
cc_library(
    name = "matrix",
    srcs = [
        "m_convert.c",
        "m_io.c",
    ],
    hdrs = [
        "include/m_convert.h",
        "include/m_io.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":dm",
        ":sm",
        ":dms",
        ":nm",
    ],
)

cc_library(
    name = "nm",
    srcs = [
        "nm.c",
    ],
    hdrs = [
        "include/nm.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    copts = [
        "-O3",
        "-ffast-math",
        "-funsafe-math-optimizations",
        "-Xpreprocessor",
        "-fopenmp"
    ],
    linkopts = [
        "-lm",
        "-lomp",
    ],
    deps = [
        ":sm",
        "@libomp"
    ],
)

# double matrix library
cc_library(
    name = "dm",
    srcs = [
        "dm.c",
    ],
    hdrs = [
        "include/dm.h",
    ],
    copts = select({
        "//conditions:default": dm_copts + ["-fopenmp"],
        ":use_accelerate": dm_copts + [
            "-DUSE_ACCELERATE",
            "-DACCELERATE_NEW_LAPACK",
        ],
        ":use_accelerate_mps": dm_copts + [
            "-DUSE_ACCELERATE",
            "-DACCELERATE_NEW_LAPACK",
        ],
        ":use_openblas": dm_copts + [
            "-DUSE_OPENBLAS",
        ],
    }),
    includes = ["include"],
    linkopts = select({
        "//conditions:default": ["-lomp"],
        ":use_accelerate": [
            "-framework Accelerate",
             "-lomp",
        ],
        ":use_accelerate_mps": [
            "-framework Accelerate",
             "-lomp",
        ],
        ":use_openblas": [
            "-L/opt/homebrew/opt/openblas/lib",
            "-lopenblas",
            "-lomp",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": default_deps + ["@libomp"],
        ":use_accelerate": default_deps + ["@libomp"],
        ":use_accelerate_mps": default_deps +["@libomp"],
        ":use_openblas": default_deps + ["@openblas", "@libomp"],
    }),
)

# single matrix library (float)
cc_library(
    name = "sm",
    srcs = [
        "sm.c",
    ],
    hdrs = [
        "include/sm.h",
    ],
    copts = select({
        "//conditions:default": sm_copts  + ["-fopenmp"],
        ":use_accelerate": sm_copts + [
            "-DUSE_ACCELERATE",
            "-DACCELERATE_NEW_LAPACK",
            "-march=native",
        ],
        ":use_accelerate_mps": sm_copts + [
            "-DUSE_ACCELERATE_MPS",
            "-DACCELERATE_NEW_LAPACK",
            "-DUSE_MPS",
            "-fobjc-arc",
        ],
        ":use_openblas": sm_copts + [
            "-DUSE_OPENBLAS",
        ],
    }),
    includes = ["include"],
    linkopts = select({
        "//conditions:default": ["-lomp"],
        ":use_accelerate": [
            "-framework Accelerate",
             "-lomp",
        ],
        ":use_accelerate_mps": [
            "-framework Accelerate",
            "-framework Metal",
            "-framework MetalPerformanceShaders", 
             "-lomp",
        ],
        ":use_openblas": [
            "-L/opt/homebrew/opt/openblas/lib",
            "-lopenblas",
            "-lomp",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": default_deps + ["@libomp"],
        ":use_accelerate": default_deps + ["@libomp"],
        ":use_accelerate_mps": default_deps + [":sm_mps_objc", "@libomp"],
        ":use_openblas": default_deps + ["@openblas", "@libomp"],
    }),
)

# Objective-C library for Metal Performance Shaders (on macOS)
objc_library(
    name = "sm_mps_objc",
    hdrs = ["sm_mps.h"],
    srcs = ["sm_mps.m"],
    copts = ["-ObjC"],
)


# sparse matrix (double) library
cc_library(
    name = "dms",
    srcs = [
        "dms.c",
    ],
    hdrs = [
        "include/dms.h",
    ],
    includes = ["include"],
    deps = default_deps + [
        "@suitesparse",
        "@libomp"
    ],
)

# build configurations for different matrix libraries

config_setting(
    name = "use_accelerate",
    values = {"define": "USE_ACCELERATE=1"},
)

config_setting(
    name = "use_openblas",
    values = {"define": "USE_OPENBLAS=1"},
)

config_setting(
    name = "use_accelerate_mps",
    values = {"define": "USE_ACCELERATE_MPS=1"},
)


