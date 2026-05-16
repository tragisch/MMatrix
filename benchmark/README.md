# Benchmark on GitHub (macOS Apple Silicon)

This folder contains a lightweight setup to run the existing cross-framework tensor benchmark
from this repository on a **self-hosted** GitHub Actions runner.

## Why this setup?

For realistic `app/tensor` vs PyTorch/MLX comparisons, runner quality matters.
A dedicated Apple Silicon machine (e.g. M2 Pro / M3 Max) gives more meaningful numbers than
shared CI resources.

## Contents

- `run_cross_framework.sh` — executes the benchmark and stores timestamped result files
- `requirements.txt` — Python deps for cross-framework run (PyTorch + MLX)
- `results/.gitkeep` — keeps result directory in Git

The benchmark implementation itself is reused from:

- `share/benchmarks/scripts/bench_conv_cross_framework.py`

## Self-hosted runner labels

The workflow runs as a matrix over two runner classes (`m2-pro`, `m3-max`).
Each machine should have:

- `self-hosted`
- `macOS`
- `ARM64`
- `benchmark`
- either `m2-pro` or `m3-max`

## Running in GitHub

1. Open **Actions** → `Benchmark Tensor on macOS (self-hosted)`.
2. Click **Run workflow** and set `repeats`.
3. Download artifacts:
	- `benchmark-results-m2-pro-<run_id>`
	- `benchmark-results-m3-max-<run_id>`

Artifacts include:

- `cross_framework_<timestamp>.txt` (raw console output)
- `cross_framework_latest.txt` (latest run shortcut)
