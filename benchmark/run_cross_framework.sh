#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="$ROOT_DIR/benchmark/results"
REPEATS="${1:-${REPEATS:-5}}"

mkdir -p "$RESULT_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_FILE="$RESULT_DIR/cross_framework_${STAMP}.txt"
LATEST_FILE="$RESULT_DIR/cross_framework_latest.txt"

{
  echo "# Benchmark run metadata"
  echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root: $ROOT_DIR"
  echo "repeats: $REPEATS"
  echo ""
  python3 "$ROOT_DIR/share/benchmarks/scripts/bench_conv_cross_framework.py" \
    --repo-root "$ROOT_DIR" \
    --repeats "$REPEATS"
} | tee "$OUT_FILE"

cp "$OUT_FILE" "$LATEST_FILE"

echo "Wrote benchmark output to: $OUT_FILE"
