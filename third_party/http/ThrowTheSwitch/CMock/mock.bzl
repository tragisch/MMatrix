# buildifier: disable=function-docstring
def mock(
        name,
        srcs,
        deps = [],
        plugins = ["ignore", "ignore_arg", "expect_any_args", "cexception", "callback", "return_thru_ptr", "array"],
        visibility = None,
        enforce_strict_ordering = False,
        strippables = [],
        treat_as_void = [],
        verbosity = 2,
        copts = [],
        unity = "@Unity//:Unity",
        linkopts = [],
        includes = [],
        when_ptr = "smart",
        fail_on_unexpected_calls = True):
    mock_srcs = name + "Srcs"
    sub_dir = __extract_sub_dir_from_header_path(srcs[0])
    other_arguments = __build_cmock_argument_string(
        enforce_strict_ordering,
        strippables,
        treat_as_void,
        verbosity,
        when_ptr,
        fail_on_unexpected_calls,
    )
    basename = __get_mock_hdr_base_name(srcs[0])
    plugin_argument = __build_plugins_argument(plugins)
    deps = __add_mock_deps(deps, plugin_argument)

    cmd = __build_mock_generator_cmd(sub_dir, plugin_argument, other_arguments)
    native.genrule(
        name = mock_srcs,
        srcs = srcs,
        outs = ["mocks/" + sub_dir + "/" + basename + ".c", "mocks/" + sub_dir + "/" + basename + ".h"],
        cmd = cmd,
        tools = [
            "@Unity//:HelperScripts",
            "@CMock//:HelperScripts",
            "@CMock//:MockGenerator",
        ],
    )
    mock_library_files = __add_header_to_srcs_if_possible([mock_srcs], srcs[0])
    native.cc_library(
        name = name,
        srcs = mock_library_files,
        hdrs = [mock_srcs],
        copts = copts,
        linkopts = linkopts,
        includes = includes,
        deps = [
            unity,
            "@CMock//:CMock",
        ] + deps,
        strip_include_prefix = "mocks/",
        visibility = visibility,
    )


def generate_a_mock_for_every_file(
        name,
        file_list,
        deps = [],
        copts = [],
        linkopts = [],
        visibility = ["//visibility:private"],
        enforce_strict_ordering = True):
    for file in file_list:
        mock(
            name = __get_mock_hdr_base_name(file),
            srcs = [file],
            deps = deps,
            copts = copts,
            linkopts = linkopts,
            visibility = visibility,
        )


def __extract_sub_dir_from_header_path(single_header_path):
    sub_dir = single_header_path
    if sub_dir.count("//") > 0:
        sub_dir = sub_dir.partition("//")[2]
    sub_dir = sub_dir.replace(":", "/").rsplit("/", 1)[0]
    if sub_dir.startswith("//"):
        sub_dir = sub_dir[2:]
    elif sub_dir.startswith("/"):
        sub_dir = sub_dir[1:]
    if sub_dir.endswith("/") and sub_dir.length > 1:
        sub_dir = sub_dir[:-1]
    return sub_dir

def __build_plugins_argument(plugins):
    plugin_argument = ""
    if len(plugins) > 0:
        plugin_argument = ";".join(plugins) + ";'"
        plugin_argument = " --plugins='" + plugin_argument
    return plugin_argument

def __get_mock_hdr_base_name(path):
    return "Mock" + path.split("/")[-1].split(":")[-1][:-2]

def __file_comes_from_current_package(file_name):
    return native.package_name() == Label(file_name).package

def __build_mock_generator_cmd(sub_dir, plugin_argument, other_arguments):
    cmd = "UNITY_DIR=external/Unity/ ruby $(location @CMock//:MockGenerator) --mock_path=$(@D)/mocks/"
    if not sub_dir == "":
        cmd = cmd + " --subdir=" + sub_dir
    cmd = cmd + plugin_argument + other_arguments + " $(SRCS)"
    return cmd

def __add_mock_deps(deps, plugin_argument):
    if plugin_argument.find("cexception") >= 0:
        deps = deps + ["@CException"]
    return deps

def __add_header_to_srcs_if_possible(srcs, header):
    if __file_comes_from_current_package(header):
        srcs = srcs + [header]
    return srcs