def _impl(ctx):
    target_dir = ctx.attr.target_dir
    files = ctx.files.srcs + ctx.files.hdrs
    dummy_output = ctx.actions.declare_file("dummy_output.txt")

    # Commands to copy files
    cmds = []
    for file in files:
        if file.path.endswith((".dylib",".a", ".h")):
            cmds.append("mkdir -p {0} && cp {1} {0}".format(target_dir, file.path))
           
    # Append a command to create a dummy output file
    cmds.append("echo 'Copy complete' > {}".format(dummy_output.path))
   
    # Create a shell command action to perform the copy and touch the dummy file
    ctx.actions.run_shell(
        inputs = files,  # Include all files being copied as inputs
        outputs = [dummy_output],  # Dummy output to satisfy Bazel's requirement
        command = "&&".join(cmds),
        progress_message = "Copying libraries and headers to {}".format(target_dir)
    )

install_library = rule(
    implementation = _impl,
    attrs = {
        "srcs": attr.label_list(allow_files=True),  # Binary files
        "hdrs": attr.label_list(allow_files=True),  # Header files
        "target_dir": attr.string(),
    }
)
