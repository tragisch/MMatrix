# Benchmark on GitHub (macOS Apple Silicon)

This folder contains a lightweight setup to run the existing cross-framework tensor benchmark
from this repository on a GitHub-hosted macOS runner.

## Why this setup?

For realistic `app/tensor` vs PyTorch/MLX comparisons, runner quality matters.
GitHub-hosted runners are easy to use and require no local runner management.

## Contents

- `run_cross_framework.sh` — executes the benchmark and stores timestamped result files
- `requirements.txt` — Python deps for cross-framework run (PyTorch + MLX)
- `results/.gitkeep` — keeps result directory in Git

The benchmark implementation itself is reused from:

- `share/benchmarks/scripts/bench_conv_cross_framework.py`

## Runner

The workflow runs on:

- `macos-latest`

## Running in GitHub

1. Open **Actions** → `Benchmark Tensor on macOS`.
2. Click **Run workflow** and set `repeats`.
3. Download artifact `benchmark-results-<run_id>`.

Artifacts include:

- `cross_framework_<timestamp>.txt` (raw console output)
- `cross_framework_latest.txt` (latest run shortcut)
