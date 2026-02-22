#!/usr/bin/env bash
set -euo pipefail

# Ensure output lands in workspace root, even when run from Bazel exec contexts.
if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
  cd "${BUILD_WORKSPACE_DIRECTORY}"
fi

# Default: only analyze app/matrix to avoid fragile third-party packages.
if [[ $# -eq 0 ]]; then
  set -- "//app/matrix:matrix"
fi

exec bazel run @wolfd_bazel_compile_commands//:generate_compile_commands -- "$@"
