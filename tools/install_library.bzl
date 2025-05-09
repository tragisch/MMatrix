def _impl(ctx):
    files = ctx.files.srcs + ctx.files.hdrs
    out_dir = ctx.actions.declare_directory("install_out")

    cmds = []
    cmds.append("mkdir -p \"$OUT_DIR\"")

    for file in files:
        if file.path.endswith((".dylib", ".a", ".h")):
            cmds.append("cp \"{src}\" \"$OUT_DIR/\"".format(src = file.path))

    cmd_lines = ["set -e", "OUT_DIR=\"{}\"".format(out_dir.path)] + cmds
    command = "\n".join(cmd_lines)

    ctx.actions.run_shell(
        inputs = files,
        outputs = [out_dir],
        command = command,
        progress_message = "Installing headers and libraries to {}".format(out_dir.path)
    )

install_library = rule(
    implementation = _impl,
    attrs = {
        "srcs": attr.label_list(allow_files=True),  # Binary files
        "hdrs": attr.label_list(allow_files=True),  # Header files
        # Zielverzeichnis ist nun festgelegt durch declare_directory()
    }
)

def install_library_macro(name, srcs = [], hdrs = []):
    install_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
)