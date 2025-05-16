load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:objc_library.bzl", "objc_library")


# define the platforms
constraint_setting(name = "custom_os")
constraint_value(
    name = "macos_arm64",
    constraint_setting = ":custom_os",
)

platform(
    name = "macos_arm64_platform",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
        ":macos_arm64",
    ],
)

config_setting(
    name = "on_macos_arm64",
    constraint_values = [":macos_arm64"],
)

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
    copts = sm_copts + ["-fopenmp"] + select({
        ":on_macos_arm64": [
            "-DUSE_ACCELERATE_MPS",
            "-DACCELERATE_NEW_LAPACK",
            "-DUSE_MPS",
            "-fobjc-arc",
        ],
        "//conditions:default": [],
    }) + select({
        ":use_openblas": [
            "-DUSE_OPENBLAS",
        ],
        "//conditions:default": [],
    }),
    includes = ["include"],
    linkopts = select({
        ":on_macos_arm64": [
            "-framework Accelerate",
            "-framework Metal",
            "-framework MetalPerformanceShaders",
            "-lomp",
        ],
        "//conditions:default": ["-lomp"],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + [":sm", "@libomp"] + select({
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    })
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
    copts = sm_copts + ["-fopenmp"] + select({
        ":on_macos_arm64": [
            "-DUSE_ACCELERATE_MPS",
            "-DACCELERATE_NEW_LAPACK",
            "-DUSE_MPS",
            "-fobjc-arc",
        ],
        "//conditions:default": [],
    }) + select({
        ":use_openblas": [
            "-DUSE_OPENBLAS",
        ],
        "//conditions:default": [],
    }),
    includes = ["include"],
    linkopts = select({
        ":on_macos_arm64": [
            "-framework Accelerate",
            "-framework Metal",
            "-framework MetalPerformanceShaders",
            "-lomp",
        ],
        "//conditions:default": ["-lomp"],
    }),
    visibility = ["//visibility:public"],
    deps = default_deps + [":sm_mps_objc", "@libomp"] + select({
        ":use_openblas": ["@openblas"],
        "//conditions:default": [],
    })
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




