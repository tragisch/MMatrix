#!/usr/bin/env bash
set -euo pipefail

workspace="${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}"
output="$(mktemp -d "${TMPDIR:-/tmp}/mmatrix-install-test.XXXXXX")"
trap 'rm -rf "$output"' EXIT

case "$(uname -s)" in
  Darwin)
    compiler=(xcrun clang)
    linkopts=(-framework Accelerate -framework Metal -framework MetalPerformanceShaders
              -framework MetalPerformanceShadersGraph -framework Foundation -lc++)
    ;;
  Linux)
    compiler=(clang)
    linkopts=(-lstdc++ -lgfortran -lquadmath -lm -lpthread -ldl)
    ;;
  *) echo "Unsupported installation test host" >&2; exit 1 ;;
esac

for mode in standard bundle; do
  prefix="$output/$mode"
  bazel run //:matrix_installer --color=no --curses=no -- --mode="$mode" "$prefix"
  "${compiler[@]}" "$workspace/tools/ci/install_consumer.c" -I"$prefix/include" \
    "$prefix"/lib/*.a "${linkopts[@]}" -o "$output/consumer-$mode"
  "$output/consumer-$mode"
done
