"""Host-aware local repository wrapper for Homebrew-installed libraries."""

def _default_path_for_os(repository_ctx):
    os_name = repository_ctx.os.name.lower()
    if "mac" in os_name:
        return repository_ctx.attr.macos_path
    if "linux" in os_name:
        return repository_ctx.attr.linux_path
    if "windows" in os_name:
        return repository_ctx.attr.windows_path
    fail("Unsupported host OS for {name}: {os}".format(
        name = repository_ctx.name,
        os = repository_ctx.os.name,
    ))

def _select_root(repository_ctx):
    env_value = repository_ctx.os.environ.get(repository_ctx.attr.env_var, "").strip()
    if env_value:
        return env_value

    root = _default_path_for_os(repository_ctx)
    if root:
        return root

    fail("No path configured for repository {name}. Set {env_var} or provide an OS default path.".format(
        name = repository_ctx.name,
        env_var = repository_ctx.attr.env_var,
    ))

def _ensure_exists(repository_ctx, path, description):
    if not repository_ctx.path(path).exists:
        fail("{description} not found for repository {name}: {path}".format(
            description = description,
            name = repository_ctx.name,
            path = path,
        ))

def _host_local_library_impl(repository_ctx):
    root = _select_root(repository_ctx)
    _ensure_exists(repository_ctx, root, "Library root")

    for subdir in repository_ctx.attr.required_subdirs:
        source = root + "/" + subdir
        _ensure_exists(repository_ctx, source, "Required subdir")
        repository_ctx.symlink(source, subdir)

    repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD.bazel")

host_local_library = repository_rule(
    implementation = _host_local_library_impl,
    local = True,
    environ = [
        "OPENBLAS_ROOT",
        "SUITESPARSE_ROOT",
    ],
    attrs = {
        "build_file": attr.label(mandatory = True, allow_single_file = True),
        "env_var": attr.string(mandatory = True),
        "linux_path": attr.string(default = ""),
        "macos_path": attr.string(default = ""),
        "required_subdirs": attr.string_list(default = []),
        "windows_path": attr.string(default = ""),
    },
)
