#!/usr/bin/env bash
set -euo pipefail

workspace="${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}"
output="$(mktemp -d "${TMPDIR:-/tmp}/mmatrix-install-test.XXXXXX")"
completed_modes=0
# Older Bash can return success after a nounset expansion error in a loop.
# Require both consumers to finish, including on the EXIT cleanup path.
trap 'status=$?; rm -rf "$output"; [[ "$completed_modes" == 2 ]] || status=1; exit "$status"' EXIT

case "$(uname -s)" in
  Darwin)
    compiler=(xcrun clang)
    linkopts=(-framework Accelerate -framework Metal -framework MetalPerformanceShaders
              -framework MetalPerformanceShadersGraph -framework Foundation -lc++)
    ;;
  Linux)
    # GNU ld resolves static archives from left to right; the dependency
    # closure is installed as separate archives rather than one link order.
    compiler=(clang -Wl,--start-group)
    linkopts=(-Wl,--end-group -lstdc++ -lgfortran -lquadmath -lm -lpthread -ldl)
    ;;
  *) echo "Unsupported installation test host" >&2; exit 1 ;;
esac

for mode in standard bundle; do
  prefix="$output/$mode"
  bazel run //:matrix_installer --color=no --curses=no -- --mode="$mode" "$prefix"
  "${compiler[@]}" "$workspace/tools/ci/install_consumer.c" -I"$prefix/include" \
    "$prefix"/lib/*.a "${linkopts[@]}" -o "$output/consumer-$mode"
  "$output/consumer-$mode"
  echo "$mode installation consumer passed"
  completed_modes=$((completed_modes + 1))
done
