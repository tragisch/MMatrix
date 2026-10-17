"""Reusable Bazel rule + macro for clang-doc generation and sync."""

load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

def _clang_doc_archive_impl(ctx):
    output_tar = ctx.outputs.out

    args = [
        ctx.file.compile_commands.path,
        output_tar.path,
        ctx.attr.filter,
        ctx.attr.format,
        ctx.attr.project_name,
        ctx.attr.source_root,
        ctx.attr.repository,
    ] + ctx.attr.extra_args

    ctx.actions.run_shell(
        inputs = [ctx.file.compile_commands],
        outputs = [output_tar],
        command = """
set -euo pipefail

COMPDB="$1"
OUT="$2"
FILTER="$3"
FORMAT="$4"
PROJECT_NAME="$5"
SOURCE_ROOT="$6"
REPOSITORY="$7"
shift 7

if ! command -v clang-doc >/dev/null 2>&1; then
  echo "clang-doc not found in PATH" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/clang-doc.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

OUT_DIR="${TMP_DIR}/docs"
mkdir -p "${OUT_DIR}"

DOC_ARGS=(
  --executor=all-TUs
  --format="${FORMAT}"
  --filter="${FILTER}"
  --output="${OUT_DIR}"
)

if [[ -n "${PROJECT_NAME}" ]]; then
  DOC_ARGS+=("--project-name=${PROJECT_NAME}")
fi

if [[ -n "${SOURCE_ROOT}" ]]; then
  DOC_ARGS+=("--source-root=${SOURCE_ROOT}")
fi

if [[ -n "${REPOSITORY}" ]]; then
  DOC_ARGS+=("--repository=${REPOSITORY}")
fi

while (($#)); do
  DOC_ARGS+=("$1")
  shift
done

clang-doc "${DOC_ARGS[@]}" "${COMPDB}"

tar -cf "${OUT}" -C "${OUT_DIR}" .
""",
        arguments = args,
        mnemonic = "ClangDoc",
        progress_message = "Generating clang-doc for {}".format(ctx.label),
        use_default_shell_env = True,
    )

    return [DefaultInfo(files = depset([output_tar]))]

clang_doc_archive = rule(
    implementation = _clang_doc_archive_impl,
    attrs = {
        "compile_commands": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "filter": attr.string(mandatory = True),
        "format": attr.string(default = "md"),
        "project_name": attr.string(default = ""),
        "source_root": attr.string(default = ""),
        "repository": attr.string(default = ""),
        "extra_args": attr.string_list(default = []),
        "out": attr.output(mandatory = True),
    },
)

def clang_doc_target(
        name,
        filter,
        out,
        default_dest,
        compile_commands = "//:compile_commands.json",
        format = "md",
        project_name = "",
        source_root = "",
        repository = "",
        extra_args = [],
        tags = [],
        target_compatible_with = None,
        visibility = None):
    """Creates clang-doc build + sync targets.

    Run target:     :<name> (bazel run ... -- [optional-dest])
    Build artifact: :<name>_archive
    """

    archive_name = name + "_archive"

    clang_doc_archive(
        name = archive_name,
        compile_commands = compile_commands,
        filter = filter,
        format = format,
        project_name = project_name,
        source_root = source_root,
        repository = repository,
        extra_args = extra_args,
        out = out,
        tags = tags,
        target_compatible_with = target_compatible_with,
        visibility = visibility,
    )

    sh_binary(
        name = name,
        srcs = ["//tools/clang_doc:sync_docs.sh"],
        data = [":" + archive_name],
        args = [
            "$(location :{})".format(archive_name),
            default_dest,
        ],
        tags = tags,
        target_compatible_with = target_compatible_with,
        visibility = visibility,
    )
