#
# Copyright (c) 2025 @tragisch <https://github.com/tragisch>
# SPDX-License-Identifier: MIT
#
# This file is part of a project licensed under the MIT License.
#  See the LICENSE file in the root directory for details.
#

load("//:tools/unity_test.bzl", "unity_test")

## Unit tests:

unity_test(
    name = "test_dm",
    srcs = ["test_dm.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_sm",
    srcs = ["test_sm.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_dms",
    srcs = ["test_dms.c",],
    my_config = ["my_config.yml"],
     deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_convert",
    srcs = ["test_convert.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

unity_test(
    name = "test_io",
    srcs = ["test_io.c",],
    my_config = ["my_config.yml"],
    deps = ["//src:matrix",],
    tags = ["smoke"],
)

