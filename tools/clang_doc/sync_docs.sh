#!/usr/bin/env bash
set -o pipefail -o errexit -o nounset

if [[ $# -lt 2 ]]; then
  echo "Usage: sync_docs.sh <tar-file> <default-dest> [dest-override]" >&2
  exit 1
fi

TAR_FILE="$1"
DEFAULT_DEST="$2"
DEST="${3:-$DEFAULT_DEST}"

# Resolve destination relative to workspace when possible.
if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" && "$DEST" != /* ]]; then
  DEST="${BUILD_WORKSPACE_DIRECTORY}/${DEST}"
fi

mkdir -p "$DEST"
rm -rf "$DEST"/*

tar -xf "$TAR_FILE" -C "$DEST"

echo "clang-doc synchronized to: $DEST"
