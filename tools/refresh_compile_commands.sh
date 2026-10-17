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

bazel run @wolfd_bazel_compile_commands//:generate_compile_commands -- "$@"

# Bazel compile_commands may contain unresolved Apple placeholders that break clangd
# system-header resolution (e.g. inttypes.h via SDK headers).
if [[ "$(uname -s)" == "Darwin" && -f "compile_commands.json" ]]; then
  sdkroot="$(xcrun --sdk macosx --show-sdk-path)"
  developer_dir="${DEVELOPER_DIR:-$(xcode-select -p)}"

  MMATRIX_SDKROOT="$sdkroot" MMATRIX_DEVELOPER_DIR="$developer_dir" python3 - <<'PY'
import pathlib

path = pathlib.Path("compile_commands.json")
text = path.read_text(encoding="utf-8")

import os
sdkroot = os.environ["MMATRIX_SDKROOT"]
developer = os.environ["MMATRIX_DEVELOPER_DIR"]

text = text.replace("__BAZEL_XCODE_SDKROOT__", sdkroot)
text = text.replace("__BAZEL_XCODE_DEVELOPER_DIR__", developer)

path.write_text(text, encoding="utf-8")
PY
fi
