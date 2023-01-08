# buildifier: disable=no-effect
""" 
Use the helper scripts shipped with unity to generate a test runner for the specified file.
"""
def generate_test_runner(file_name, name = None, visibility = None, cexception = True):
    if cexception:
        cmd = "ruby $(location @Unity//:TestRunnerGenerator) -cexception $(SRCS) $(OUTS)"
    else:
        cmd = "ruby $(location @Unity//:TestRunnerGenerator)  $(SRCS) $(OUTS)"
    if name == None:
        name = runner_base_name(file_name)
    native.genrule(
        name = name,
        srcs = [file_name],
        outs = [name + ".c"],
        cmd = cmd,
        tools = [
            "@Unity//:TestRunnerGenerator",
            "@Unity//:HelperScripts",
        ],
        visibility = visibility,
    )


# buildifier: disable=no-effect
""" This macro creates a cc_test rule and a genrule (that creates the test runner) for a given file.

It adds unity as dependency so the user doesn't have to do it himself.
Additional dependencies can be specified using the deps parameter.

The source files for the test are only the *_Test.c that the user writes
and the corresponding generated *_Test_Runner.c file.
"""
def unity_test(
        file_name,
        name = [],
        deps = [],
        copts = [],
        size = "small",
        cexception = True,
        linkopts = [],
        visibility = None,
        unity = "@Unity//:Unity",
        additional_srcs = []):
    generate_test_runner(
        file_name,
        visibility,
        cexception = cexception,
    )
    native.cc_test(
        name = strip_extension(file_name),
        srcs = [file_name, runner_file_name(file_name)] + additional_srcs,
        visibility = visibility,
        deps = deps + [unity],
        size = size,
        linkopts = linkopts,
        copts = copts,
        linkstatic = 1,
    )


# i guess this could be done more elegant using the File class but it does what we want
def strip_extension(file_name):
    return file_name[0:-2]

def runner_base_name(file_name):
    return strip_extension(file_name) + "_Runner"

def runner_file_name(file_name):
    return runner_base_name(file_name) + ".c"
