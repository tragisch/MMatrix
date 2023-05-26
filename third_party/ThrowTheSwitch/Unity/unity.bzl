def unity_test(name,  srcs, deps=None,  **kwargs):
    """ Create a unity test.

    Args:
        name: The name of the test.
        srcs: The source files of the test.
        deps: The dependencies of the test.
        **kwargs: Additional arguments to pass to the cc_test rule.
    """
    deps =(deps or []) + [Label("@Unity//:Unity")]
    file_name = str(srcs[0])
    generate_test_runner(file_name)

    native.cc_test(
        name = name,
        deps = deps,
        srcs = srcs + [runner_file_name(file_name)],
        size = "small",
        visibility = ["//visibility:public"],
        **kwargs
    )


def generate_test_runner(file_name, name=None):
    """ Generate a test runner for a given test file.

    Args:
        file_name: The name of the test file.
        name: The name of the generated test runner.
    """
    cmd = "ruby $(location @Unity//:TestRunnerGenerator)  $(SRCS) $(OUTS)"
    out_name = runner_file_name(file_name)
    if name == None:
        name = runner_base_name(file_name)
    native.genrule(
        name = name,
        srcs = [file_name],    
        outs = [out_name],
        cmd = cmd,
        tools = [
            "@Unity//:TestRunnerGenerator",
            "@Unity//:HelperScripts",
        ],
        visibility = ["//visibility:public"],
    )

def strip_extension(file_name):
    return file_name[0:-2]

def runner_base_name(file_name):
    return str(strip_extension(file_name)) + "_Runner"

def runner_file_name(file_name):
    return str(runner_base_name(file_name)) + ".c"