#!/usr/bin/env python3
"""Cross-framework Conv2D benchmark on Apple Silicon.

Compares:
- C backend (this repo): forced GEMM and forced MPS via bench_st_ab_conv
- PyTorch MPS (torch.nn.functional.conv2d)
- MLX (mlx.nn.Conv2d)

Focus shapes are aligned with app/tensor benchmark cases.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ConvShape:
    name: str
    n: int
    c_in: int
    c_out: int
    h: int
    w: int
    k: int
    stride: int
    pad: int
    warmup: int
    iters: int


SHAPES: Tuple[ConvShape, ...] = (
    ConvShape("conv_medium", 4, 32, 64, 56, 56, 3, 1, 1, 3, 10),
    ConvShape("conv_large", 8, 64, 128, 112, 112, 3, 1, 1, 2, 5),
    ConvShape("pw_medium", 4, 64, 128, 56, 56, 1, 1, 0, 3, 10),
)


def run(cmd: List[str], cwd: str) -> str:
    out = subprocess.check_output(cmd, cwd=cwd, text=True)
    return out


def median_ms(samples: List[float]) -> float:
    return statistics.median(samples)


def gmacs(shape: ConvShape) -> float:
    # MACs = N * Cout * Hout * Wout * Cin * K * K
    # for stride=1, pad chosen to keep spatial for K=3 in these cases
    hout = shape.h
    wout = shape.w
    if shape.k == 1 and shape.pad == 0:
        hout = shape.h
        wout = shape.w
    macs = shape.n * shape.c_out * hout * wout * shape.c_in * shape.k * shape.k
    return macs / 1e9


def throughput_gmacs_per_s(shape: ConvShape, ms: float) -> float:
    sec = ms / 1000.0
    return gmacs(shape) / sec


def benchmark_c_ab(repo_root: str, repeats: int) -> Dict[Tuple[str, str], float]:
    # Build and run existing C A/B benchmark (median over repeats).
    run(["bazel", "build", "-c", "opt", "//app/tensor:bench_st_ab_conv"], cwd=repo_root)
    exe = os.path.join(repo_root, "bazel-out/darwin_arm64-opt/bin/app/tensor/bench_st_ab_conv")
    # key: (case_name, variant) -> list[ms_per_iter]
    samples: Dict[Tuple[str, str], List[float]] = {}
    for _ in range(repeats):
        out = run([exe], cwd=repo_root)
        for line in out.splitlines():
            if not line.startswith("conv_ab,"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            case_name = parts[1]
            variant = parts[2]  # gemm | mps
            ms = float(parts[7])
            samples.setdefault((case_name, variant), []).append(ms)

    result: Dict[Tuple[str, str], float] = {}
    for key, arr in samples.items():
        result[key] = median_ms(arr)
    return result


def benchmark_torch_mps(shapes: Tuple[ConvShape, ...], repeats: int) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available on this machine.")

    torch.manual_seed(42)
    device = torch.device("mps")

    medians: Dict[str, float] = {}
    for s in shapes:
        x = torch.randn((s.n, s.c_in, s.h, s.w), device=device, dtype=torch.float32)
        w = torch.randn((s.c_out, s.c_in, s.k, s.k), device=device, dtype=torch.float32)

        run_times: List[float] = []
        for _ in range(repeats):
            with torch.inference_mode():
                for _ in range(s.warmup):
                    _ = F.conv2d(x, w, bias=None, stride=s.stride, padding=s.pad)
                torch.mps.synchronize()

                t0 = time.perf_counter()
                for _ in range(s.iters):
                    _ = F.conv2d(x, w, bias=None, stride=s.stride, padding=s.pad)
                torch.mps.synchronize()
                t1 = time.perf_counter()

            run_times.append((t1 - t0) * 1000.0 / s.iters)

        medians[s.name] = median_ms(run_times)
    return medians


def benchmark_mlx(shapes: Tuple[ConvShape, ...], repeats: int) -> Dict[str, float]:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(42)
    medians: Dict[str, float] = {}

    for s in shapes:
        # MLX Conv2d expects NHWC input layout.
        x = mx.random.normal((s.n, s.h, s.w, s.c_in), dtype=mx.float32)
        conv = nn.Conv2d(
            in_channels=s.c_in,
            out_channels=s.c_out,
            kernel_size=s.k,
            stride=s.stride,
            padding=s.pad,
            bias=False,
        )

        run_times: List[float] = []
        for _ in range(repeats):
            for _ in range(s.warmup):
                y = conv(x)
                mx.eval(y)

            t0 = time.perf_counter()
            for _ in range(s.iters):
                y = conv(x)
                # MLX is lazy; force materialization each iteration so we
                # measure true per-iteration latency instead of last-op-only.
                mx.eval(y)
            t1 = time.perf_counter()

            run_times.append((t1 - t0) * 1000.0 / s.iters)

        medians[s.name] = median_ms(run_times)
    return medians


def hw_snapshot() -> Dict[str, str]:
    out: Dict[str, str] = {}
    out["platform"] = platform.platform()
    try:
        out["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        out["cpu"] = "n/a"
    try:
        out["mem_bytes"] = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
    except Exception:
        out["mem_bytes"] = "n/a"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    print("# Hardware")
    for k, v in hw_snapshot().items():
        print(f"{k}: {v}")

    print("\n# Running C benchmark (forced GEMM/MPS)")
    c_rows = benchmark_c_ab(args.repo_root, args.repeats)

    print("# Running PyTorch MPS benchmark")
    torch_rows = benchmark_torch_mps(SHAPES, args.repeats)

    print("# Running MLX benchmark")
    mlx_rows = benchmark_mlx(SHAPES, args.repeats)

    print("\nframework,case,median_ms,GMAC/s")
    for s in SHAPES:
        c_gemm = c_rows.get((s.name, "gemm"))
        c_mps = c_rows.get((s.name, "mps"))
        t_ms = torch_rows.get(s.name)
        m_ms = mlx_rows.get(s.name)

        if c_gemm is not None:
            print(f"c_gemm,{s.name},{c_gemm:.6f},{throughput_gmacs_per_s(s, c_gemm):.2f}")
        if c_mps is not None:
            print(f"c_mps,{s.name},{c_mps:.6f},{throughput_gmacs_per_s(s, c_mps):.2f}")
        if t_ms is not None:
            print(f"pytorch_mps,{s.name},{t_ms:.6f},{throughput_gmacs_per_s(s, t_ms):.2f}")
        if m_ms is not None:
            print(f"mlx,{s.name},{m_ms:.6f},{throughput_gmacs_per_s(s, m_ms):.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
