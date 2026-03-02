#!/usr/bin/env python3
"""Import benchmark text files into a SQLite database for Grafana.

Supported inputs:
- share/simple_benchmark/benchmark_results.txt
- data/results_perf.txt

The importer is idempotent per input line (line hash with INSERT OR IGNORE).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


TIME_TOKEN_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ns|us|ms|s)")
MEM_TOKEN_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>KB|MB|GB)")
SIZE_RE = re.compile(r"\(([^)]*)\)")


@dataclass
class Record:
    source_file: str
    benchmark_name: str
    size_label: str | None
    backend: str | None
    time_ns: int | None
    memory_bytes: int | None
    density: float | None
    layout_type: int | None
    os_name: str | None
    host_name: str | None
    measured_at: str | None
    raw_time: str | None
    raw_memory: str | None
    raw_line: str
    line_hash: str


def parse_time_to_ns(time_text: str) -> int | None:
    t = time_text.strip()
    if not t:
        return None

    # Handle ctime-like "2m 54s" format.
    if "m" in t and "s" in t:
        m = re.search(r"(\d+)\s*m", t)
        s = re.search(r"(\d+)\s*s", t)
        if m and s:
            minutes = int(m.group(1))
            seconds = int(s.group(1))
            return (minutes * 60 + seconds) * 1_000_000_000

    m = TIME_TOKEN_RE.fullmatch(t)
    if not m:
        # Already numeric seconds (e.g. results_perf)
        try:
            return int(float(t) * 1_000_000_000)
        except ValueError:
            return None

    value = float(m.group("value"))
    unit = m.group("unit")
    if unit == "ns":
        return int(value)
    if unit == "us":
        return int(value * 1_000)
    if unit == "ms":
        return int(value * 1_000_000)
    if unit == "s":
        return int(value * 1_000_000_000)
    return None


def parse_memory_to_bytes(memory_text: str) -> int | None:
    m = MEM_TOKEN_RE.fullmatch(memory_text.strip())
    if not m:
        return None
    value = float(m.group("value"))
    unit = m.group("unit")
    if unit == "KB":
        return int(value * 1024)
    if unit == "MB":
        return int(value * 1024 * 1024)
    if unit == "GB":
        return int(value * 1024 * 1024 * 1024)
    return None


def parse_ctime_like(date_text: str) -> str | None:
    value = date_text.strip()
    if not value:
        return None
    # ctime() style, e.g. "Thu May  1 15:02:15 2025"
    try:
        dt = datetime.strptime(value, "%a %b %d %H:%M:%S %Y")
        return dt.isoformat(sep=" ")
    except ValueError:
        return None


def make_hash(source_file: str, raw_line: str) -> str:
    return hashlib.sha1(f"{source_file}|{raw_line}".encode("utf-8")).hexdigest()


def infer_size_label(benchmark_name: str) -> str | None:
    m = SIZE_RE.search(benchmark_name)
    if not m:
        return None
    return m.group(1).strip()


def parse_simple_benchmark_file(path: Path) -> Iterable[Record]:
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("function"):
            continue

        parts = [p.strip() for p in raw.split("\t") if p.strip()]
        if len(parts) < 5:
            continue

        # New format from write_double_to_file:
        # function_name, time_usage, memory_usage, comment/backend, os, host, ctime
        if len(parts) >= 7:
            benchmark_name = parts[0]
            raw_time = parts[1]
            raw_memory = parts[2]
            backend = parts[3]
            os_name = parts[4]
            host_name = parts[5]
            date_text = " ".join(parts[6:])
            measured_at = parse_ctime_like(date_text)
        else:
            # Legacy format in existing file:
            # Function, Time, Memory, Date, Comments
            benchmark_name = parts[0]
            raw_time = parts[1]
            raw_memory = parts[2]
            measured_at = parse_ctime_like(parts[3])
            backend = parts[4]
            os_name = None
            host_name = None

        yield Record(
            source_file=str(path),
            benchmark_name=benchmark_name,
            size_label=infer_size_label(benchmark_name),
            backend=backend,
            time_ns=parse_time_to_ns(raw_time),
            memory_bytes=parse_memory_to_bytes(raw_memory),
            density=None,
            layout_type=None,
            os_name=os_name,
            host_name=host_name,
            measured_at=measured_at,
            raw_time=raw_time,
            raw_memory=raw_memory,
            raw_line=raw,
            line_hash=make_hash(str(path), raw),
        )


def parse_results_perf_file(path: Path) -> Iterable[Record]:
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("func"):
            continue

        parts = [p.strip() for p in raw.split("\t") if p.strip()]
        if len(parts) < 5:
            continue

        benchmark_name = parts[0]
        raw_time = parts[1]
        density = None
        layout_type = None
        try:
            density = float(parts[2])
        except ValueError:
            pass
        try:
            layout_type = int(parts[3])
        except ValueError:
            pass

        date_text = " ".join(parts[4:])
        measured_at = parse_ctime_like(date_text)

        backend = f"layout_type={layout_type}" if layout_type is not None else None

        yield Record(
            source_file=str(path),
            benchmark_name=benchmark_name,
            size_label=infer_size_label(benchmark_name),
            backend=backend,
            time_ns=parse_time_to_ns(raw_time),
            memory_bytes=None,
            density=density,
            layout_type=layout_type,
            os_name=None,
            host_name=None,
            measured_at=measured_at,
            raw_time=raw_time,
            raw_memory=None,
            raw_line=raw,
            line_hash=make_hash(str(path), raw),
        )


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA busy_timeout=5000;
        PRAGMA journal_mode=DELETE;

        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            benchmark_name TEXT NOT NULL,
            size_label TEXT,
            backend TEXT,
            time_ns INTEGER,
            memory_bytes INTEGER,
            density REAL,
            layout_type INTEGER,
            os_name TEXT,
            host_name TEXT,
            measured_at TEXT,
            raw_time TEXT,
            raw_memory TEXT,
            raw_line TEXT NOT NULL,
            line_hash TEXT NOT NULL UNIQUE,
            inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_benchmark_runs_name_time
            ON benchmark_runs (benchmark_name, measured_at);

        CREATE INDEX IF NOT EXISTS idx_benchmark_runs_backend
            ON benchmark_runs (backend);

        CREATE VIEW IF NOT EXISTS benchmark_runs_v AS
        SELECT
            id,
            source_file,
            benchmark_name,
            size_label,
            backend,
            time_ns,
            time_ns / 1e6 AS time_ms,
            memory_bytes,
            memory_bytes / 1024.0 / 1024.0 AS memory_mb,
            density,
            layout_type,
            os_name,
            host_name,
            measured_at,
            raw_time,
            raw_memory,
            inserted_at
        FROM benchmark_runs;
        """
    )


def insert_records(conn: sqlite3.Connection, records: Iterable[Record]) -> int:
    inserted = 0
    for r in records:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO benchmark_runs (
                source_file,
                benchmark_name,
                size_label,
                backend,
                time_ns,
                memory_bytes,
                density,
                layout_type,
                os_name,
                host_name,
                measured_at,
                raw_time,
                raw_memory,
                raw_line,
                line_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.source_file,
                r.benchmark_name,
                r.size_label,
                r.backend,
                r.time_ns,
                r.memory_bytes,
                r.density,
                r.layout_type,
                r.os_name,
                r.host_name,
                r.measured_at,
                r.raw_time,
                r.raw_memory,
                r.raw_line,
                r.line_hash,
            ),
        )
        inserted += cur.rowcount
    return inserted


def parse_file(path: Path) -> Iterable[Record]:
    name = path.name.lower()
    if name == "benchmark_results.txt":
        return parse_simple_benchmark_file(path)
    if name == "results_perf.txt":
        return parse_results_perf_file(path)
    # Heuristic fallback
    if "results_perf" in name:
        return parse_results_perf_file(path)
    return parse_simple_benchmark_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import benchmark text files into SQLite")
    parser.add_argument(
        "--db",
        default="data/benchmarks.db",
        help="Path to sqlite database (default: data/benchmarks.db)",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "share/simple_benchmark/benchmark_results.txt",
            "data/results_perf.txt",
        ],
        help="Input files to parse",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    total_inserted = 0
    total_seen = 0

    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)

        for input_path_str in args.inputs:
            path = Path(input_path_str)
            if not path.exists():
                print(f"[skip] missing input: {path}")
                continue

            records = list(parse_file(path))
            total_seen += len(records)
            inserted = insert_records(conn, records)
            total_inserted += inserted
            print(f"[ok] {path}: parsed={len(records)} inserted={inserted}")

        conn.commit()

    print(
        f"Done. db={db_path} parsed_total={total_seen} inserted_total={total_inserted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
