load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
  name = "matio",
  srcs = glob(["src/*.c", "src/*.h"]),
  hdrs = glob(["src/*.h"])+["matioConfig.h", "matio_pubconf.h"],
  includes = ["src"],
 # linkopts = ["-lz", "-lhdf5"],
  deps = [
    "@zlib",
   # "@hdf5",
    ":config_h",
  ],
  visibility = ["//visibility:public"],
)

genrule(
    name = "config_h",
    srcs = glob(["**"]),  # Include all files in the library's source
    outs = [
       "matioConfig.h", "matio_pubconf.h"
    ],
    cmd = """
./$(location configure)
cp src/matioConfig.h $(@D)/matioConfig.h
cp src/matio_pubconf.h $(@D)/matio_pubconf.h
""",
    tools = ["configure"],
    # deps = [
    #     "@hdf5",
    # ],
    visibility = ["//visibility:public"],
)

# genrule(
#     name = "matioConfig_h",
#     srcs = ["@matio//:visual_studio/matioConfig.h" ],
#     outs = ["matioConfig.h"],
#     cmd = "cp $< $@",
# )

# genrule(
#     name = "pubconf_h",
#     srcs = ["@matio//:visual_studio/matio_pubconf.h"],
#     outs = ["matio_pubconf.h"],
#     cmd = "cp $< $@",
# )
